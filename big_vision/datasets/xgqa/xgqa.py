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
"""Generates xGQA in a TFDS-ready structure.

First, download the data:
  mkdir -p /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_bn.json -P /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_de.json -P /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_en.json -P /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_id.json -P /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_ko.json -P /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_pt.json -P /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_ru.json -P /tmp/data/xgqa/annotations
  wget https://raw.githubusercontent.com/e-bug/iglue/main/datasets/xGQA/annotations/zero_shot/testdev_balanced_questions_zh.json -P /tmp/data/xgqa/annotations
  wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -P /tmp/data/xgqa/
  unzip /tmp/data/xgqa/images.zip -d /tmp/data/xgqa/

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=xgqa

Example to load:

import tensorflow_datasets as tfds
dataset = tfds.load(
    'xgqa', split='test_zs_en',
    data_dir='/tmp/tfds')
"""
import json
import os

import tensorflow_datasets as tfds

_DESCRIPTION = """xGQA (uses GQA images)."""

# pylint: disable=line-too-long
_CITATION = (
    '@inproceedings{pfeiffer-etal-2022-xgqa,'
    'title = "x{GQA}: Cross-Lingual Visual Question Answering",'
    'author = "Pfeiffer, Jonas  and'
    '  Geigle, Gregor  and'
    '  Kamath, Aishwarya  and'
    '  Steitz, Jan-Martin  and'
    '  Roth, Stefan  and'
    '  Vuli{\'c}, Ivan  and'
    '  Gurevych, Iryna",'
    'booktitle = "Findings of the Association for Computational Linguistics: '
    'ACL 2022",'
    'month = may,'
    'year = "2022",'
    'address = "Dublin, Ireland",'
    'publisher = "Association for Computational Linguistics",'
    'url = "https://aclanthology.org/2022.findings-acl.196",'
    'doi = "10.18653/v1/2022.findings-acl.196",'
    'pages = "2497--2511",'
    '}'
)
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_DATA_PATH = '/tmp/data/xgqa/'
_IMAGE_PATH = '/tmp/data/xgqa/images/'

LANGUAGES = frozenset(['bn', 'de', 'en', 'id', 'ko', 'pt', 'ru', 'zh'])


class XGQA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for XGQA dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'example_id': tfds.features.Text(),
            'image/id': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'question': tfds.features.Text(),
            'answer': tfds.features.Text(),
        }),
        supervised_keys=None,
        homepage='https://github.com/adapter-hub/xGQA',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    d = dict()
    for l in LANGUAGES:
      d.update({
          f'test_zs_{l}': self._generate_examples('test', 'zero_shot', l),
          f'test_fs_{l}': self._generate_examples('test', 'few_shot', l),
          f'dev_fs_{l}': self._generate_examples('test', 'few_shot', l),
          f'train_fs1_{l}': self._generate_examples('train_1', 'few_shot', l),
          f'train_fs5_{l}': self._generate_examples('train_5', 'few_shot', l),
          f'train_fs10_{l}': self._generate_examples('train_10', 'few_shot', l),
          f'train_fs20_{l}': self._generate_examples('train_20', 'few_shot', l),
          f'train_fs25_{l}': self._generate_examples('train_25', 'few_shot', l),
          f'train_fs48_{l}': self._generate_examples('train_48', 'few_shot', l),
      })
    return d

  def _generate_examples(self, split, num_shots, lang):
    """Yields (key, example) tuples."""
    # Loads the questions for each image.
    if num_shots == 'few_shot':
      file_path = os.path.join(_DATA_PATH, 'annotations', 'few_shot', lang,
                               f'{split}.json')
    elif num_shots == 'zero_shot':
      file_path = os.path.join(_DATA_PATH, 'annotations', 'zero_shot',
                               f'testdev_balanced_questions_{lang}.json')
    else:
      raise ValueError(f'Unknown num_shots: {num_shots}')
    with open(file_path, 'r') as f:
      entries = json.load(f)

    # Make one entry per question-answer pair.
    for question_id, question_data in entries.items():
      example_id = f'{question_id}_{lang}'
      yield example_id, {
          'example_id': example_id,
          'image/id': question_data['imageId'],
          'image': os.path.join(_IMAGE_PATH, f'{question_data["imageId"]}.jpg'),
          'question': question_data['question'],
          'answer': question_data['answer'],
      }
