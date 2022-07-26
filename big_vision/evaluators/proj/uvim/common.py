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

"""Common utilities used in evaluators."""
import math
import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def get_jax_process_dataset(dataset, split, global_batch_size, pp_fn,
                            dataset_dir=None, cache=True, add_tfds_id=False):
  """Returns dataset to be processed by current jax host.

  The dataset is sharded and padded with zeros such that all processes
  have equal number of batches. The first 2 dimensions of the dataset
  elements are: [local_device_count, device_batch_size].

  Args:
    dataset: dataset name.
    split: dataset split.
    global_batch_size: batch size to be process per iteration on the dataset.
    pp_fn: preprocessing function to apply per example.
    dataset_dir: path for tfds to find the prepared data.
    cache: whether to cache the dataset after batching.
    add_tfds_id: whether to add the unique `tfds_id` string to each example.
  """
  assert global_batch_size % jax.device_count() == 0
  total_examples = tfds.load(
      dataset, split=split, data_dir=dataset_dir).cardinality()
  num_batches = math.ceil(total_examples / global_batch_size)

  process_split = tfds.even_splits(
      split, n=jax.process_count(), drop_remainder=False)[jax.process_index()]
  data = tfds.load(
      dataset,
      split=process_split,
      data_dir=dataset_dir,
      read_config=tfds.ReadConfig(add_tfds_id=add_tfds_id)).map(pp_fn)
  pad_data = tf.data.Dataset.from_tensors(
      jax.tree_map(lambda x: tf.zeros(x.shape, x.dtype), data.element_spec)
  ).repeat()

  data = data.concatenate(pad_data)
  data = data.batch(global_batch_size // jax.device_count())
  data = data.batch(jax.local_device_count())
  data = data.take(num_batches)
  if cache:
    # Eval datasets are often used many times and caching the dataset after
    # batching allows one to have the buffers ready to be used and not have
    # to wait for preprocessing to be done over and over.
    data = data.cache()
  return data
