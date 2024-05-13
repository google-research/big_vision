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

# pylint: disable=line-too-long
r"""Implements textcaps val-set in TFDS structure.

It's small data, so simple to run locally. First, copy the data to local disk:

  mkdir -p /tmp/data/textcaps
  cd /tmp/data/textcaps
  curl -O https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json
  curl -O https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json
  curl -O https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_test.json
  curl -O https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
  curl -O https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
  unzip train_val_images.zip
  rm train_val_images.zip
  unzip test_images.zip
  rm test_images.zip

Then, run conversion locally (make sure to install tensorflow-datasets for the
`tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=textcaps


Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('text_caps', split='val', data_dir='/tmp/tfds')
"""
import collections
import json
import os

from absl import logging
import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """TextCaps dataset."""

# pylint: disable=line-too-long
_CITATION = (
    '@inproceedings{sidorov2019textcaps,'
    'title={TextCaps: a Dataset for Image Captioningwith Reading Comprehension},'
    'author={Sidorov, Oleksii and Hu, Ronghang and Rohrbach, Marcus and Singh, Amanpreet},'
    'journal={European Conference on Computer Vision},'
    'year={2020}}')
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_FILEPATH = '/tmp/data/textcaps/'
_TRAIN_FILES = '/tmp/data/textcaps/TextCaps_0.1_train.json'
_VAL_FILES = '/tmp/data/textcaps/TextCaps_0.1_val.json'
_TEST_FILES = '/tmp/data/textcaps/TextCaps_0.1_test.json'


class TextCaps(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for TextCaps dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata.

    (tfds.core.DatasetInfo object)
      These are the features of your dataset like images, labels, etc.
    """

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image/id': tfds.features.Text(),
            'image_filepath': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'texts': tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://textvqa.org/textcaps/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    def group_by_id(data, image_dir):
      id_to_example = collections.defaultdict(list)
      for ex in data:
        id_to_example[ex['image_id']].append(ex)

      for k, exs in id_to_example.items():
        image_ids, image_names, texts = [], [], []
        for ex in exs:
          image_ids.append(ex['image_id'])
          image_names.append(ex['image_name'])
          if ex.get('caption_str'):
            texts.append(ex.get('caption_str'))
        assert len(set(image_ids)) == 1
        assert len(set(image_names)) == 1
        image_filepath = os.path.join(
            _FILEPATH, image_dir, str(image_names[0])+'.jpg')
        id_to_example[k] = {
            'image/id': image_ids[0],
            'image_filepath': image_filepath,
            'image': image_filepath,
            'texts': texts,
        }
      return id_to_example

    # Returns the Dict[split names, Iterator[Key, Example]]
    with open(_TRAIN_FILES) as f:
      train_data = group_by_id(json.load(f)['data'], 'train_images')
    with open(_VAL_FILES) as f:
      val_data = group_by_id(json.load(f)['data'], 'train_images')
    with open(_TEST_FILES) as f:
      test_data = group_by_id(json.load(f)['data'], 'test_images')
    return {
        'train': self._generate_examples(train_data),
        'val': self._generate_examples(val_data),
        'test': self._generate_examples(test_data),
    }

  def _generate_examples(self, data):
    """Generate a tf.Example object.

    This contains the image, objects, attributes, regions and relationships.

    Args:
      data: a dictionary with the image/id.

    Yields:
      (key, example) tuples from dataset. The example has format specified in
        the above DatasetInfo.
    """
    for k, v in data.items():
      yield k, v
