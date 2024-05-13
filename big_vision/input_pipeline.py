# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet input pipeline."""
import collections
import functools
import itertools
import math
import multiprocessing.pool

from absl import logging
from big_vision.datasets import sequence_packing
import big_vision.datasets.core as ds_core
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import einops
import jax
import numpy as np
import tensorflow as tf


DEFAULT_NUM_PARALLEL_CALLS = 100


def make_for_train(
    data, preprocess_fn, batch_size,
    shuffle_buffer_size=None, cache_raw=False,
    num_parallel_calls=DEFAULT_NUM_PARALLEL_CALLS, prefetch=2,
    *,
    pre_filter_fn=None, post_filter_fn=None,
    pack=None, skip_errors=False,
):
  """Makes an input pipeline for training."""
  # Use data filtering at your own risk: the actual split sizes won't be known
  # in advance, so epoch-based things won't work correctly.

  data = _add_tpu_host_options(data)

  data = data.filter(pre_filter_fn) if pre_filter_fn else data
  data = data.cache() if cache_raw else data

  # First shuffle and then repeat (each with a different shuffle). This way
  # the data for one epoch is all seen before the next one is processed and
  # significantly affects the number of times each example is seen when
  # processing for small number of epochs.
  if shuffle_buffer_size:
    data = data.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
  data = data.repeat(None)

  data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
  data = data.filter(post_filter_fn) if post_filter_fn else data

  data = data.ignore_errors(log_warning=True) if skip_errors else data

  data = sequence_packing.pack_dataset(data, pack) if pack else data

  # Drop remainder makes shape fully static, so we can later use it if needed.
  if batch_size:
    data = data.batch(batch_size // jax.process_count(), drop_remainder=True)
  if prefetch:  # None means autotune, but we never want that.
    data = data.prefetch(prefetch)
  return data


def training(input_config):
  """Reads the data from a single dataset, or mixes it from multiple.

  The data is read either from one or mixed from multiple datasets, depending
  on the `input_config`.

  Args:
    input_config: Configures the input pipeline. See input_pipeline_test for
      examples.

  Returns:
    A tuple containing (possibly mixed) tf.data.Dataset and a total number of
    training examples.
  """
  per_pipeline_configs = (
      "shuffle_buffer_size", "cache_raw", "num_parallel_calls",
      "pre_filter_fn", "post_filter_fn", "pack", "skip_errors")
  def config_to_kw(config):
    assert "filter_fn" not in config, "Deprecated; use `pre_filter_fn` instead."
    return {k: config[k] for k in per_pipeline_configs if k in config}

  batch_size = input_config.batch_size
  # Handle separately the common case when no mixing happens.
  if isinstance(input_config.data.get("name"), str):
    train_data = ds_core.get(**input_config.data)
    train_ds = make_for_train(
        data=train_data.get_tfdata(ordered=False),
        batch_size=batch_size,
        preprocess_fn=pp_builder.get_preprocess_fn(input_config.get("pp")),
        prefetch=input_config.get("prefetch", 2),  # Default 2 for bwd compat.
        **config_to_kw(input_config)
    )
    return train_ds, train_data.total_examples

  # A helpful error instead of silent ignore:
  for k in per_pipeline_configs:
    assert k not in input_config, f"{k} is per-dataset in multi-input."

  # Parallelize the loading of datasets when doing data mixture.
  # For larger mixes, we sometimes spend >5min when doing sequentially.
  # NOTE: functools.cache is thread-safe.
  def _make(name_and_weight):
    name, weight = name_and_weight
    dataset = input_config[name]
    train_data = ds_core.get(**dataset.data)
    dataset = make_for_train(
        data=train_data.get_tfdata(ordered=False),
        # Don't batch the data just yet, it will be done after
        # mixing the different datasets below.
        batch_size=None,
        preprocess_fn=pp_builder.get_preprocess_fn(dataset.get("pp"), name),
        prefetch=0,  # Prefetching each pipeline leads to huge OOMs.
        **config_to_kw(dataset)
    )
    if keys := input_config.get("keep_only"):
      dataset = dataset.map(lambda d, keys=keys: {k: d[k] for k in keys})
    return name, dataset, weight, train_data.total_examples

  names, datasets, weights, totals = [], [], [], []
  pool = multiprocessing.pool.ThreadPool(len(input_config.data))
  for name, dataset, weight, total in pool.map(
      # Skip weight=0 datasets as a convenient optimization in sweeps.
      _make, ((name, w) for name, w in input_config.data.items() if w)):
    names.append(name)
    datasets.append(dataset)
    weights.append(weight)
    totals.append(total)

  # Normalize the weights such that they sum up to 1.
  weights = [x / sum(weights) for x in weights]

  logging.info(
      "NOTE: Total dataset mix size: %d\nContributions:\n%s", sum(totals),
      "\n".join(f"{ds}: {n} ({w * 100:.1g}%)"
                for ds, n, w in zip(names, totals, weights))
  )

  train_ds = tf.data.Dataset.sample_from_datasets(
      datasets, weights, stop_on_empty_dataset=True)
  if input_config.get("pack"):
    train_ds = sequence_packing.pack_dataset(train_ds, input_config.get("pack"))
  train_ds = train_ds.batch(
      input_config["batch_size"] // jax.process_count(), drop_remainder=True)
  if (pf := input_config.get("prefetch", 2)):
    train_ds = train_ds.prefetch(pf)

  return train_ds, sum(totals)


# The pipeline below is used for evals in multi-{G,T}PU and multi-host settings.
# As the total number of examples may not be evenly divisible accross all
# devices, we use the `infinite tf.data padding` trick, which was suggested by
# Andreas Steiner and also implemented by him in the clu library:
# https://github.com/google/CommonLoopUtils/blob/84b777c42dfd3fb6685537138433bfeb5241a006/clu/deterministic_data.py#L304.
def make_for_inference(
    data, preprocess_fn, batch_size, num_ex_per_process,
    cache_raw=False, cache_final=False,
    num_parallel_calls=DEFAULT_NUM_PARALLEL_CALLS, prefetch=1,
):
  """Makes an input pipeline for inference."""

  data = _add_tpu_host_options(data)
  data = data.cache() if cache_raw else data
  data = data.map(_add_internal_fields(preprocess_fn),
                  num_parallel_calls=num_parallel_calls)
  data = data.concatenate(_get_pad_data(data))

  local_batch_size = batch_size // jax.process_count()
  # This is just like `batch`, but allows batching elements of different shapes
  # into a tf.RaggedTensor. Elements of the same fixed shape remain tf.Tensors.
  # Since we do 'infinite' padding it is safe to drop the remainder.
  data = data.ragged_batch(batch_size=local_batch_size, drop_remainder=True)

  # We need to make sure that all hosts process all data and exactly the same
  # number of batches. Below we take max per-host num examples and use it on all
  # hosts to derive the number of batches.
  num_batches = math.ceil(max(num_ex_per_process) / local_batch_size)
  data = data.take(num_batches)

  # Note we cache data after a finite number of batches is taken.
  data = data.cache() if cache_final else data
  data = data.repeat()
  data = data.prefetch(prefetch) if prefetch else data
  return data, num_batches


def _get_pad_data(data):
  def zeros_like_spec(spec):
    # For unknown/flexible dimensions (None), just use 0 instead.
    return tf.zeros([x or 0 for x in spec.shape], spec.dtype)

  zero = jax.tree.map(zeros_like_spec, data.element_spec)
  return tf.data.Dataset.from_tensors(zero).repeat()


def _add_internal_fields(pp_fn):
  """Wraps pp_fn to add _mask and _id keys."""
  # Adds internal keys, that we either, in this order of preference:
  # 1. keep from result of pp_fn,
  # 2. carry over from raw (not pp_fn'd) example, or
  # 3. add, if that makes sense.
  def _pp_fn(example):
    result = pp_fn(example)
    # _mask will be False on padded examples (see _get_pad_data).
    result.setdefault("_mask", example.get("_mask", tf.constant(True)))
    # Not all data-sources can provide an ID. Only carry-over if it can:
    if "_id" in example and "_id" not in result:
      result["_id"] = example["_id"]
    return result
  return _pp_fn


def _add_tpu_host_options(data):
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1

  # Stop a whole bunch of magic stuff that eats up all RAM:
  options.experimental_optimization.inject_prefetch = False

  return data.with_options(options)


def prefetch_iterator(it, n):
  """Runs iterator `it` ahead for `n` steps. Adapted from flax."""
  if not n:
    yield from it
    return
  queue = collections.deque()

  def enqueue(n_steps):  # Enqueues *up to* `n` elements from the iterator.
    for data in itertools.islice(it, n_steps):
      queue.append(data)

  enqueue(n)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)


def threadstart_iterator(it):
  """Starts an iterator right away in a background thread."""
  # We already want to "start" the iterator in order to start the underlying
  # dataset prefetch mechanisms, so here we get the first element. But we don't
  # want to lose it from training, so we yield that one afterwards.
  # (internal link)
  pool = multiprocessing.pool.ThreadPool(processes=1)
  first_ex_promise = pool.apply_async(lambda: next(it))

  yield first_ex_promise.get()
  yield from it


def tf_to_numpy(x):
  """Convert any TF types to numpy."""
  if isinstance(x, tf.Tensor):
    if x.dtype != tf.string:  # Dense, non-string tensor? Easy!
      return x.numpy()
    else:  # A dense string tensor? Turn into actual strings, not bytes.
      return np.vectorize(bytes.decode, otypes=[str])(x.numpy())

  # The rest deals with RaggedTensors, for two main reasons:
  # - For strings, recursively apply the above conversion
  # - For common cases (eg batch of images), return more reasonable shapes.

  # Replace all None's in the shape by a fixed number, in the (somewhat common)
  # case that they are marked ragged, but really all have the same shape.
  real_shape = list(x.shape)
  for i, s in enumerate(real_shape[1:]):
    if s is not None: continue
    rowlens = np.diff(x.nested_row_splits[i])
    if len(set(rowlens)) == 1:
      real_shape[i + 1] = rowlens[0]

  if None not in real_shape:
    return tf_to_numpy(x.flat_values).reshape(real_shape)

  # It's actually ragged, reconstruct the array from the variable length pieces.
  splits = x.row_splits.numpy()
  rows = [tf_to_numpy(x.values[splits[i]:splits[i + 1]])
          for i in range(len(splits) - 1)]
  return np.fromiter(rows, dtype=object)


# Note that the order of global devices for sharding data is important and
# should be compatible with device order used for models params, state, etc.
def start_global(
    data, global_devices, n_prefetch=1, keep_on_cpu=frozenset(), warmup=False):
  """Starts the global input pipeline."""
  def maybe_shard(name, x):
    if name in keep_on_cpu:
      return tf_to_numpy(x)
    return u.make_fsarray_from_local_slice(x, global_devices)

  it = iter(data)
  if warmup:  # actually pre-fill shuffle buffers etc.
    it = threadstart_iterator(it)

  it = (u.tree_map_with_names(maybe_shard, elem) for elem in it)
  return prefetch_iterator(it, n_prefetch)


##########################################################################
# The code below is pmap-specific and is deprecated, please switch to jit.
##########################################################################


def shard_and_put(x, shard=True, put=True):
  x = np.asarray(memoryview(x))  # No-copy conversion: http://(internal link)
  if shard:
    x = einops.rearrange(x, "(d l) ... -> d l ...", d=jax.local_device_count())
  if shard and put:  # Only works for pmap (for now).
    x = jax.device_put_sharded(list(x), jax.local_devices())
  return x


def start_input_pipeline(data, n_prefetch=1, shard=True):
  fn = functools.partial(shard_and_put, shard=shard, put=n_prefetch)
  it = (jax.tree.map(fn, elem) for elem in iter(data))
  return prefetch_iterator(it, n_prefetch)


def start_ragged_input_pipeline(data, n_prefetch=1, shard=True, ragged=None):
  def maybe_shard_and_put(name, x):
    return x if name in (ragged or {}) else shard_and_put(x, shard)

  it = (u.tree_map_with_names(maybe_shard_and_put, elem) for elem in iter(data))
  return prefetch_iterator(it, n_prefetch)
