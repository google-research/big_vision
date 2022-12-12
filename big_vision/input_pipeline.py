# Copyright 2022 Big Vision Authors.
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

import big_vision.datasets.core as ds_core
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import einops
import flax.jax_utils as flax_utils
import jax
import tensorflow as tf


def make_for_train(
    data, preprocess_fn, batch_size,
    shuffle_buffer_size, cache_raw=False, filter_fn=None,
    num_parallel_calls=100, prefetch=2):
  """Makes an input pipeline for training."""

  data = _add_tpu_host_options(data)

  # Use data filtering at your own risk: the actual split sizes won't be known
  # in advance, so many things can go wrong in the code.
  if filter_fn:
    data = data.filter(filter_fn)

  data = data.cache() if cache_raw else data
  data = data.repeat(None)  # repeat data indefinitely
  data = data.shuffle(shuffle_buffer_size) if shuffle_buffer_size else data

  data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
  # Drop remainder makes shape fully static, so we can later use it if needed.
  if batch_size:
    data = data.batch(batch_size // jax.process_count(), drop_remainder=True)
  return data.prefetch(prefetch)


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

  batch_size = input_config.batch_size
  # Handle separately the common case when no mixing happens.
  if isinstance(input_config.data.get("name"), str):
    train_data = ds_core.get(**input_config.data)
    train_ds = make_for_train(
        data=train_data.get_tfdata(ordered=False),
        batch_size=batch_size,
        preprocess_fn=pp_builder.get_preprocess_fn(input_config.get("pp")),
        shuffle_buffer_size=input_config.get("shuffle_buffer_size"),
        cache_raw=input_config.get("cache_raw", False),
        filter_fn=input_config.get("filter_fn"),
    )
    return train_ds, train_data.total_examples

  datasets = []
  weights = []
  ntraining_examples = 0

  for dataset_name, weight in input_config.data.items():
    dataset = input_config[dataset_name]
    train_data = ds_core.get(**dataset.data)
    ntraining_examples += train_data.total_examples
    dataset = make_for_train(
        data=train_data.get_tfdata(ordered=False),
        # Don't batch the data just yet, it will be done after
        # mixing the different datasets below.
        batch_size=None,
        preprocess_fn=pp_builder.get_preprocess_fn(dataset.get("pp")),
        shuffle_buffer_size=dataset.get("shuffle_buffer_size"),
        cache_raw=dataset.get("cache_raw", False),
        filter_fn=dataset.get("filter_fn"),
    )
    datasets.append(dataset)
    weights.append(weight)

  # Normalize the weights such that they sum up to 1.
  weights = [x / sum(weights) for x in weights]

  train_ds = tf.data.Dataset.sample_from_datasets(
      datasets, weights, stop_on_empty_dataset=True)
  train_ds = train_ds.batch(
      input_config["batch_size"] // jax.process_count(), drop_remainder=True)

  return train_ds, ntraining_examples


# The pipeline below is used for evals in multi-{G,T}PU and multi-host settings.
# As the total number of examples may not be evenly divisible accross all
# devices, we use the `infinite tf.data padding` trick, which was suggested by
# Andreas Steiner and also implemented by him in the clu library:
# https://github.com/google/CommonLoopUtils/blob/84b777c42dfd3fb6685537138433bfeb5241a006/clu/deterministic_data.py#L304.
def make_for_inference(
    data, preprocess_fn, batch_size, num_ex_per_process,
    cache_raw=False, cache_final=False):
  """Makes an input pipeline for inference."""

  data = _add_tpu_host_options(data)
  data = data.cache() if cache_raw else data
  data = data.map(_add_mask(preprocess_fn), num_parallel_calls=100)
  data = data.concatenate(_get_pad_data(data))

  local_batch_size = batch_size // jax.process_count()
  # This is just like `batch`, but allows batching elements of different shapes
  # into a tf.RaggedTensor. Elements of the same fixed shape remain tf.Tensors.
  # Since we do 'infinite' padding it is safe to drop the remainder.
  data = data.apply(tf.data.experimental.dense_to_ragged_batch(
      batch_size=local_batch_size, drop_remainder=True))

  # We need to make sure that all hosts process all data and exactly the same
  # number of batches. Below we take max per-host num examples and use it on all
  # hosts to derive the number of batches.
  num_batches = math.ceil(max(num_ex_per_process) / local_batch_size)
  data = data.take(num_batches)

  # Note we cache data after a finite number of batches is taken.
  data = data.cache() if cache_final else data
  data = data.repeat()
  return data.prefetch(1), num_batches


def _get_pad_data(data):
  def zeros_like_spec(spec):
    # For unknown/flexible dimensions (None), just use 0 instead.
    return tf.zeros([x or 0 for x in spec.shape], spec.dtype)

  zero = jax.tree_map(zeros_like_spec, data.element_spec)
  return tf.data.Dataset.from_tensors(zero).repeat()


def _add_mask(pp_fn):
  def _pp_fn(example):
    return {"_mask": tf.constant(1), **pp_fn(example)}
  return _pp_fn


def _add_tpu_host_options(data):
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1
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


def shard_and_put(x, shard=True, put=True):
  # pylint: disable=protected-access
  x = x._numpy()  # avoids redundant copy when converting tf tensors to numpy.
  if shard:
    x = einops.rearrange(x, "(d l) ... -> d l ...", d=jax.local_device_count())
  if shard and put:  # Only works for pmap (for now).
    x = jax.device_put_sharded(list(x), flax_utils._pmap_device_order())
  return x
  # pylint: enable=protected-access


def start_input_pipeline(data, n_prefetch=1, shard=True):
  fn = functools.partial(shard_and_put, shard=shard, put=n_prefetch)
  it = (jax.tree_util.tree_map(fn, elem) for elem in iter(data))
  return prefetch_iterator(it, n_prefetch)


def start_ragged_input_pipeline(data, n_prefetch=1, shard=True, ragged=None):
  def maybe_shard_and_put(name, x):
    return x if name in (ragged or {}) else shard_and_put(x, shard)

  it = (u.tree_map_with_names(maybe_shard_and_put, elem) for elem in iter(data))
  return prefetch_iterator(it, n_prefetch)
