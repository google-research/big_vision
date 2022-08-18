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
import math

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
    data, preprocess_fn, batch_size, num_ex_per_process,
    cache_raw=False, cache_final=False):
  """Makes an input pipeline for inference."""

  data = _add_tpu_host_options(data)
  data = data.cache() if cache_raw else data
  data = data.map(_add_mask(preprocess_fn), num_parallel_calls=100)
  data = data.concatenate(_get_pad_data(data))

  local_batch_size = batch_size // jax.process_count()
  # Since we do 'infinite' padding it is safe to drop the remainder.
  data = data.batch(local_batch_size, drop_remainder=True)

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
