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

"""TensorFlow Datasets as data source for big_vision."""
import functools

import big_vision.datasets.core as ds_core
import jax
import numpy as np
import overrides
import tensorflow as tf
import tensorflow_datasets as tfds


class DataSource(ds_core.DataSource):
  """Use TFDS as a data source."""

  def __init__(self, name, split, data_dir=None, skip_decode=("image",)):
    self.builder = _get_builder(name, data_dir)
    self.split = split
    # Each host is responsible for a fixed subset of data
    process_splits = tfds.even_splits(split, jax.process_count())
    self.process_split = process_splits[jax.process_index()]
    self.skip_decode = skip_decode

  @overrides.overrides
  def get_tfdata(
      self, ordered=False, *, process_split=True, allow_cache=True, **kw):
    # The tf.data may use a lot of RAM, so we need to expose the option of not
    # keeping this in memory when we use lots of input pipelines, such as when
    # having many ephemeral evaluators.
    return (_cached_get_dataset if allow_cache else _get_dataset)(
        self.builder, self.skip_decode,
        split=self.process_split if process_split else self.split,
        shuffle_files=not ordered,
        **kw)

  @property
  @overrides.overrides
  def total_examples(self):
    return self.builder.info.splits[self.split].num_examples

  @overrides.overrides
  def num_examples_per_process(self):
    splits = tfds.even_splits(self.split, jax.process_count())
    return [self.builder.info.splits[s].num_examples for s in splits]


@functools.cache
def _get_builder(dataset, data_dir):
  if dataset == "from_data_dir":
    return tfds.builder_from_directory(data_dir)
  else:
    return tfds.builder(dataset, data_dir=data_dir, try_gcs=True)


# Cache as it may well take 1-2min on large datasets, and we may use the same
# multiple times (eg various evaluators).
def _get_dataset(builder, skip_decode, **kw):
  """Returns a tf.data to be used."""
  rckw = {k: kw.pop(k) for k in ("shuffle_seed",) if k in kw}
  ds = builder.as_dataset(
      read_config=tfds.ReadConfig(
          skip_prefetch=True,  # We prefetch after pipeline.
          try_autocache=False,  # We control this, esp. for few-shot.
          add_tfds_id=True,
          **rckw,
      ),
      decoders={
          f: tfds.decode.SkipDecoding()
          for f in skip_decode if f in builder.info.features
      },
      **kw)

  def _hash_tfds_id(example):
    id_ = tf.strings.to_hash_bucket_strong(
        example["tfds_id"],
        np.iinfo(np.uint32).max,  # Max value
        [3714561454027272724, 8800639020734831960])  # Magic.
    example["_id"] = tf.bitcast(id_, tf.int32)[0]  # good device dtype.
    return example

  return ds.map(_hash_tfds_id)
_cached_get_dataset = functools.cache(_get_dataset)
