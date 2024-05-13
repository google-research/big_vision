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
r"""Implements textvqa in TFDS structure.

It's small data, so simple to run locally. First, copy the data to local disk:

  mkdir -p /tmp/data/textvqa
  cd /tmp/data/textvqa
  curl -O https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
  curl -O https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
  curl -O https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
  curl -O https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
  curl -O https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json
  # The Rosetta_OCR files are probably not needed.
  # curl -O https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json
  # curl -O https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json
  # curl -O https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_test.json
  unzip train_val_images.zip
  rm train_val_images.zip
  unzip test_images.zip
  rm test_images.zip
  # Background: at https://textvqa.org/dataset/ it says:
  # "Note: Some of the images in OpenImages are rotated,
  # please make sure to check the Rotation field in the Image IDs files
  # for train and test."
  curl -O https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
  curl -O https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
  mv train-images-boxable-with-rotation.csv train_images/rotation.csv
  mv test-images-with-rotation.csv test_images/rotation.csv

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=textvqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('textvqa', split='train', data_dir='/tmp/tfds')
"""
import json
import os

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


_DESCRIPTION = """TextVqa dataset."""

# pylint: disable=line-too-long
_CITATION = (
    '@inproceedings{singh2019towards,'
    'title={Towards VQA Models That Can Read},'
    'author={Singh, Amanpreet and Natarjan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Parikh, Devi and Rohrbach, Marcus},'
    'booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},'
    'pages={8317-8326},'
    'year={2019}}'
    )
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above and use these:
_FILEPATH = '/tmp/data/textvqa/'
_TRAIN_FILES = '/tmp/data/textvqa/TextVQA_0.5.1_train.json'
_VAL_FILES = '/tmp/data/textvqa/TextVQA_0.5.1_val.json'
_TEST_FILES = '/tmp/data/textvqa/TextVQA_0.5.1_test.json'
_ROTATION_CSV = 'rotation.csv'


class TextVqa(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for textvqa dataset."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Undo rotation for known rotated images.',
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
            'image/id': tfds.features.Scalar(np.int32),
            'image_filepath': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'question_id': tfds.features.Scalar(np.int32),
            'question': tfds.features.Text(),
            'answers': tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://textvqa.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    def json_to_examples(data, image_dir):
      # Load rotation csv.
      logging.info('Processing %d items in %s', len(data), image_dir)
      rot = pd.read_csv(os.path.join(_FILEPATH, image_dir, _ROTATION_CSV))
      rotation_by_id = {}
      for row in rot.itertuples():
        rotation = int(row.Rotation) if not np.isnan(row.Rotation) else 0
        rotation_by_id[row.ImageID] = rotation

      examples = {}
      for v in data:
        image_id = str(v['image_id'])
        image_filepath = os.path.join(_FILEPATH, image_dir, image_id + '.jpg')
        question_id = v['question_id']
        examples[question_id] = {
            'image/id': question_id,
            'image_filepath': image_filepath,
            'image': image_filepath,
            'rotation': rotation_by_id[image_id],
            'question_id': question_id,
            'question': v['question'],
            'answers': v.get('answers', []),  # No answers in test set.
        }
      return examples

    # Returns the Dict[split names, Iterator[Key, Example]]
    with open(_TRAIN_FILES) as f:
      train_data = json_to_examples(json.load(f)['data'], 'train_images')
    with open(_VAL_FILES) as f:
      # Validation images are stored in the train_images folder.
      val_data = json_to_examples(json.load(f)['data'], 'train_images')
    with open(_TEST_FILES) as f:
      test_data = json_to_examples(json.load(f)['data'], 'test_images')
    return {
        'train': self._generate_examples(train_data),
        'val': self._generate_examples(val_data),
        'test': self._generate_examples(test_data),
    }

  def _generate_examples(self, data):
    """Generate a tf.Example object.

    Args:
      data: a dictionary with the image/id.

    Yields:
      (key, example) tuples from dataset. The example has format specified in
        the above DatasetInfo.
    """
    for k, v in data.items():
      # If the image is rotated, we undo the rotation here and re-encode.
      image_bytes = open(v['image_filepath'], 'rb').read()
      if v['rotation'] != 0:
        rotation = v['rotation']
        assert rotation % 90 == 0
        turns = int(rotation / 90)
        image = tf.image.decode_jpeg(image_bytes)
        image_bytes = tf.io.encode_jpeg(
            tf.image.rot90(image, turns), quality=100
        ).numpy()
      # If no rotation was needed, we just pass along the unchanged bytes.
      v['image'] = image_bytes

      # Now all rotation should have been accounted for. And we don't want to
      # pass on the (now obsolete) rotation info as features.
      del v['rotation']

      yield k, v
