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
r"""Implements RSVQA-HR dataset in TFDS.

Remote sensing visual question answering task, using high-resolution airborne
image data at 15cm resolution per pixel.

It's small dataset at source (14G), so simple to run locally.
First, download and unzip the dataset from https://zenodo.org/records/6344367
and place it in /tmp/data/rsvqa_hr.

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd third_party/py/big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=rsvqa_hr

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('rsvqa_hr', split='train', data_dir='/tmp/tfds')

Dataset splits (all):
  train: 625,340 examples/questions
  val: 102,843 examples/questions
  test: 222,684 examples/questions
  test_2: 105,647 examples/questions  (other area, unknown instrument)
Non-numeric data splits (nonum):
  train: 371,834 examples/questions
  val: 60,405 examples/questions
  test: 131,468 examples/questions
  test_2: 62,554 examples/questions

Note: due to image duplication with each question, the dataset size is
significatnly increased by the number of questions per image.

Recommended training splits:
  train: train
  minitrain: train[:5%]
  eval: val
  full_train: train+val
  test: test

Image sizes: 512x512
Number of answers per question: 1
Question types distribution in train split:
  - Area (area): 14.6% (integers, binned into {0m2, 1-10m2, 11-100m2, 101-1000m2, >1000m2})
  - Comparison(comp): 33.5%
  - Count (count): 26.0%  (integers, not binned, maximum number of objects is 89)
  - Presence (presence): 26.0%
"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """RSVQA-HR dataset."""

# pylint: disable=line-too-long
_CITATION = """
@article{Lobry_2020,
   title={RSVQA: Visual Question Answering for Remote Sensing Data},
   volume={58},
   ISSN={1558-0644},
   url={http://dx.doi.org/10.1109/TGRS.2020.2988782},
   DOI={10.1109/tgrs.2020.2988782},
   number={12},
   journal={IEEE Transactions on Geoscience and Remote Sensing},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Lobry, Sylvain and Marcos, Diego and Murray, Jesse and Tuia, Devis},
   year={2020},
   month=dec, pages={8555-8566} }
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
PATH = '/tmp/data/rsvqa_hr/'


class RsvqaHrConfig(tfds.core.BuilderConfig):
  """Config to specify each variant."""

  def __init__(self, nonum, **kwargs):
    name = 'nonum' if nonum else 'all'
    super(RsvqaHrConfig, self).__init__(name=name, **kwargs)
    self.nonum = nonum


class RsvqaHr(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for RSVQA-HR dataset."""

  VERSION = tfds.core.Version('1.0.2')
  RELEASE_NOTES = {
      '1.0.0': 'First release.',
      '1.0.1': 'Rename binned values.',
      '1.0.2': 'Removed explicit png image encoding.',
      }

  BUILDER_CONFIGS = [
      RsvqaHrConfig(nonum=False),
      RsvqaHrConfig(nonum=True),
  ]

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'question_id': tfds.features.Scalar(np.int32),
            'filename': tfds.features.Text(),
            'image': tfds.features.Image(),
            'question': tfds.features.Text(),
            'question_type': tfds.features.Text(),
            'answers': tfds.features.Sequence(tfds.features.Text()),
            'raw_answers': tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,
        homepage='https://rsvqa.sylvainlobry.com/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
        split: self._generate_examples(split)
        for split in ('train', 'val', 'test', 'test_2')
    }

  def _generate_examples(self, split):
    """Yields (key, example) tuples."""
    if split == 'test_2':
      split = 'test_phili'
    questions_path = os.path.join(PATH + f'USGS_split_{split}_questions.json')
    answers_path = os.path.join(PATH + f'USGS_split_{split}_answers.json')
    images_path = os.path.join(PATH + 'Data')

    with open(questions_path, 'r') as f:
      questions = json.loads(f.read())['questions']
    with open(answers_path, 'r') as f:
      answers = json.loads(f.read())['answers']

    for q, a in zip(questions, answers):
      assert q['active'] == a['active']
      if not q['active']:
        continue
      if self.builder_config.nonum and q['type'] in ('area', 'count'):
        continue
      assert q['answers_ids'][0] == a['id']
      assert q['id'] == a['question_id']

      filename = f'{q["img_id"]}.png'
      yield q['id'], {
          'question_id': q['id'],
          'filename': filename,
          'image': os.path.join(images_path, filename),
          'question': q['question'],
          'question_type': q['type'],
          'answers': [bin_answer(a['answer'], q['type'])],
          'raw_answers': [a['answer']],
      }


def bin_answer(answer, question_type):
  """Bins answers into expected ranges."""
  if question_type == 'area':
    area = int(answer[:-2])
    if area == 0:
      return '0 m2'
    elif area <= 10:
      return 'between 1 m2 and 10 m2'
    elif area <= 100:
      return 'between 11 m2 and 100 m2'
    elif area <= 1000:
      return 'between 101 m2 and 1000 m2'
    else:
      return 'more than 1000 m2'
  return answer
