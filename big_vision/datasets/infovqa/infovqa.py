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
r"""Implements InfoVqa in TFDS structure.

First, download and unzip the dataset from https://rrc.cvc.uab.es/?ch=17
and place it in /tmp/data/infovqa.

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd third_party/py/big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=infovqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('infovqa', split='train', data_dir='/tmp/tfds')

Dataset splits:
  train: 23946 examples/questions (4406 images)
  val: 2801 examples/questions (500 images)
  test: 3288 examples/questions (579 images) (no answers)

Recommended training splits:
  train: train[:95%] (22749 examples/questions)
  minitrain: train[:5%] (1197 examples/questions)
  minival: train[95%:] (1197 examples/questions)
  eval: val (2801 examples/questions)

Note that according to task description in
https://rrc.cvc.uab.es/?ch=17&com=tasks:
  - Order of items in a multi span answer does not matter. Therefore, we include
    all permutations of the answer in the val split.
  - Answers are not case sensitive. We leave it to the user to lower case
    answers if they want to.
"""
import itertools
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """InfographicVQA dataset."""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{Mathew_2022,
   title={InfographicVQA},
   url={http://dx.doi.org/10.1109/WACV51458.2022.00264},
   DOI={10.1109/wacv51458.2022.00264},
   booktitle={2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
   publisher={IEEE},
   author={Mathew, Minesh and Bagal, Viraj and Tito, Ruben and Karatzas, Dimosthenis and Valveny, Ernest and Jawahar, C. V.},
   year={2022},
   month=jan }
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_INFOVQA_PATH = '/tmp/data/infovqa/'
_ANNOTATIONS = {
    'train': 'infographicsVQA_train_v1.0.json',
    'val': 'infographicsVQA_val_v1.0_withQT.json',
    'test': 'infographicsVQA_test_v1.0.json',
    }


class Infovqa(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for infovqa dataset."""

  VERSION = tfds.core.Version('1.1.0')
  RELEASE_NOTES = {
      '1.0.0': 'First release.',
      '1.1.0': 'Add multi-span permutations to the val split answers.',
      }

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'question_id': tfds.features.Scalar(np.int32),
            'filename': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'question': tfds.features.Text(),
            'answers': tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,
        homepage='https://www.docvqa.org/datasets/infographicvqa',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split)
            for split in ('train', 'val', 'test')}

  def _generate_examples(self, split):
    """Yields (key, example) tuples from test set."""
    annot_fname = os.path.join(_INFOVQA_PATH, _ANNOTATIONS[split])
    with open(annot_fname, 'r') as f:
      data = json.loads(f.read())

    for x in data['data']:
      yield x['questionId'], {
          'question_id': x['questionId'],
          'filename': x['image_local_name'],
          'image': os.path.join(_INFOVQA_PATH, 'images', x['image_local_name']),
          'question': x['question'],
          'answers': maybe_permute(x.get('answers', []), split),
      }


def maybe_permute(answers, split):
  if split != 'val':
    return answers
  new_answers = []
  for x in answers:
    if ', ' in x:  # Create all permutations.
      # The first element remains the same.
      new_answers.extend([', '.join(y)
                          for y in itertools.permutations(x.split(', '))])
    else:
      new_answers.append(x)
  return new_answers
