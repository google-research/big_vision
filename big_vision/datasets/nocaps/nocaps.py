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
r"""Implements nocaps val/test set in TFDS structure.

It's small data, so simple to run locally. First, copy the data to local disk:

  mkdir -p /tmp/data/nocaps_data
  cd /tmp/data/nocaps_data
  wget https://s3.amazonaws.com/open-images-dataset/tar/test.tar.gz
  wget https://s3.amazonaws.com/open-images-dataset/tar/validation.tar.gz
  curl -O https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
  curl -O https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json

  mkdir -p /tmp/data/nocaps_data/Images
  tar -xf validation.tar.gz -C Images
  rm validation.tar.gz
  tar -xf test.tar.gz -C Images
  rm test.tar.gz

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=nocaps

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('nocaps', split='val', data_dir='/tmp/tfds')
"""
import collections
import json
import os

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


_DESCRIPTION = """Nocaps dataset."""

_CITATION = (
    '@inproceedings{agrawal2019nocaps,'
    'title={nocaps: novel object captioning at scale},'
    'author={Agrawal, Harsh and Desai, Karan and Wang, Yufei and Chen, Xinlei'
    'and Jain, Rishabh and Johnson, Mark and Batra, Dhruv and Parikh, Devi'
    'and Lee, Stefan and Anderson, Peter},'
    'booktitle={ICCV},'
    'pages={8948--8957},'
    'year={2019}}')

# When running locally (recommended), copy files as above an use these:
_FILEPATH = '/tmp/data/nocaps_data/Images/'
_VAL_FILES = '/tmp/data/nocaps_data/nocaps_val_4500_captions.json'
_TEST_FILES = '/tmp/data/nocaps_data/nocaps_test_image_info.json'


class NoCaps(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for nocaps dataset."""

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
            'image/id': tf.int64,
            'image_filepath': tfds.features.Text(),
            'url': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'texts': tfds.features.Sequence(tfds.features.Text()),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://nocaps.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    def group_by_id(data, image_dir):
      id2caps = collections.defaultdict(list)
      for ex in data.get('annotations', []):
        id2caps[ex['image_id']].append(ex['caption'])

      id_to_example = {}
      for ex in data['images']:
        id_to_example[ex['id']] = {
            'image/id': ex['id'],
            'image_filepath': os.path.join(
                _FILEPATH, image_dir, ex['file_name']),
            'url': ex['coco_url'],
            'image': os.path.join(_FILEPATH, image_dir, ex['file_name']),
            'texts': id2caps[ex['id']] if ex['id'] in id2caps else ['N/A'],
        }
      return id_to_example

    # Returns the Dict[split names, Iterator[Key, Example]]
    with open(_VAL_FILES) as f:
      val_data = group_by_id(json.load(f), 'validation')
    with open(_TEST_FILES) as f:
      test_data = group_by_id(json.load(f), 'test')
    return {
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
      try:
        # Jpeg decode test to check early errors. The decoded images are not
        # used, instead we rely on the default tfds.features.Image function.
        unused_image = tf.io.read_file(v['image_filepath'])
        unused_image = np.array(tf.image.decode_jpeg(unused_image))
      except tf.errors.InvalidArgumentError:
        # Unable to read image, skip this image and output download link.
        logging.error('Unable to decode: curl -O %s', v['url'])
        continue
      except tf.errors.NotFoundError:
        # Unable to read image, skip this image and output download link.
        logging.error('File not found: curl -O %s', v['url'])
        continue

      yield k, v
