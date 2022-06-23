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
import functools
import math
import einops
import flax.jax_utils as flax_utils
import jax

import tensorflow as tf
import tensorflow_datasets as tfds


@functools.lru_cache(maxsize=None)
def get_builder(dataset, data_dir):
  return tfds.builder(dataset, data_dir=data_dir, try_gcs=True)


def get_num_examples(dataset, split, data_dir=None):
  builder = get_builder(dataset, data_dir)
  return builder.info.splits[split].num_examples


def get_max_examples_per_host(dataset, split, data_dir=None):
  """Returns the max number of examples accross all hosts."""
  splits = tfds.even_splits(split, jax.process_count())
  return max([get_num_examples(dataset, s, data_dir) for s in splits])


def get_dataset_tfds(dataset="imagenet2012", split="train",
                     shuffle_files=True, data_dir=None, skip_decode=("image",)):
  """Data provider."""
  builder = get_builder(dataset, data_dir)
  split = tfds.even_splits(split, jax.process_count())[jax.process_index()]
  skip_decoders = {
      f: tfds.decode.SkipDecoding()
      for f in skip_decode
      if f in builder.info.features
  }
  # Each host is responsible for a fixed subset of data
  return builder.as_dataset(
      split=split,
      shuffle_files=shuffle_files,
      read_config=tfds.ReadConfig(
          skip_prefetch=True,  # We prefetch after pipeline.
          try_autocache=False,  # We control this, esp. for few-shot.
          add_tfds_id=True,
      ),
      decoders=skip_decoders), builder


def make_for_train(
    dataset, split, preprocess_fn, batch_size,
    shuffle_buffer_size, cache_raw=False, data_dir=None, filter_fn=None,
    num_parallel_calls=100, prefetch=2):
  """Makes an input pipeline for training."""

  data, _ = get_dataset_tfds(dataset=dataset, split=split,
                             shuffle_files=True, data_dir=data_dir)
  data = _add_tpu_host_options(data)

  # Use data filtering at your own risk: the actual split sizes won't be known
  # in advance, so many things can go wrong in the code.
  if filter_fn:
    data = data.filter(filter_fn)

  data = data.cache() if cache_raw else data
  data = data.repeat(None)  # repeat data indefinetely
  data = data.shuffle(shuffle_buffer_size) if shuffle_buffer_size else data

  data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
  # Drop remainder makes shape fully static, so we can later use it if needed.
  data = data.batch(batch_size // jax.process_count(), drop_remainder=True)
  return data.prefetch(prefetch)


# The pipeline below is used for evals in multi-{G,T}PU and multi-host settings.
# As the total number of examples may not be evenly divisible accross all
# devices, we use the `infinite tf.data padding` trick, which was suggested by
# Andreas Steiner and also implemented by him in the clu library:
# https://github.com/google/CommonLoopUtils/blob/84b777c42dfd3fb6685537138433bfeb5241a006/clu/deterministic_data.py#L304.
def make_for_inference(
    dataset, split, preprocess_fn, batch_size, data_dir=None,
    cache_raw=False, cache_final=False):
  """Makes an input pipeline for inference."""
  data, _ = get_dataset_tfds(dataset=dataset, split=split,
                             shuffle_files=False, data_dir=data_dir)
  data = _add_tpu_host_options(data)
  data = data.cache() if cache_raw else data
  data = data.map(_add_mask(preprocess_fn), num_parallel_calls=100)
  data = data.concatenate(_get_pad_data(data))
  # Since we do 'infinite' padding it is safe to drop the remainder.
  data = data.batch(batch_size // jax.process_count(), drop_remainder=True)

  # We need to make sure that all hosts process all data and exactly the same
  # number of batches. Below we take max per-host num examples and use it on all
  # hosts to derive the number of batches.
  n = get_max_examples_per_host(dataset, split, data_dir)
  num_batches = math.ceil(n / (batch_size // jax.process_count()))
  data = data.take(num_batches)

  # Note we cache data after a finite number of batches is taken.
  data = data.cache() if cache_final else data
  data = data.repeat()
  return data.prefetch(1), num_batches


def _get_pad_data(data):
  zero = jax.tree_map(lambda x: tf.zeros(x.shape, x.dtype), data.element_spec)
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


def shard_fn(x):
  return einops.rearrange(x, "(d l) ... -> d l ...", d=jax.local_device_count())


def start_input_pipeline(data, n_prefetch, shard=True):
  # _numpy() call avoids redundant copy when converting tf tensors to numpy.
  it = (jax.tree_map(lambda x: x._numpy(), b) for b in iter(data))  # pylint: disable=protected-access
  if shard:
    it = (jax.tree_map(shard_fn, b) for b in it)
  if shard and n_prefetch:  # Only works for pmap.
    it = flax_utils.prefetch_to_device(it, n_prefetch)
  return it


