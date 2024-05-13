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
r"""Implements RSVQA-LR dataset in TFDS.

Remote sensing visual question answering task, using low-resolution satellite
(Sentinel-2) RGB channels data at 10m resolution per pixel.

It's small dataset at source (200M), so simple to run locally.
First, download and unzip the dataset from https://zenodo.org/records/6344334
and place it in /tmp/data/rsvqa_lr.

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd third_party/py/big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=rsvqa_lr

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('rsvqa_lr', split='train', data_dir='/tmp/tfds')

Dataset splits:
  train: 57223 examples/questions
  val: 10005 examples/questions
  test: 10004 examples/questions
And the same splits are available excluding numeric questions:
  train_nonum: 39441 examples/questions
  val_nonum: 6782 examples/questions
  test_nonum: 6782 examples/questions

Note: due to image duplication with each question, the dataset size is
significatnly increased by the number of questions per image.

Recommended training splits:
  train: train
  minitrain: train[:5%]
  eval: val
  full_train: train+val
  test: test

Image sizes: 256x256
Number of answers per question: 1
Question types distribution in train split:
  - Comparison(comp): 39.4%
  - Count (count): 29.9%  (integers, binned at evaluation into
    {0, 1-10, 11-100, 101-1000, >10000})
  - Presence (presence): 29.7%
  - Rural/Urban (rural_urban): 1%
"""
import io
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """RSVQA-LR dataset."""

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
   month=dec, pages={8555–8566} }
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
PATH = '/tmp/data/rsvqa_lr/'


class RsvqaLrConfig(tfds.core.BuilderConfig):
  """Config to specify each variant."""

  def __init__(self, nonum, **kwargs):
    name = 'nonum' if nonum else 'all'
    super(RsvqaLrConfig, self).__init__(name=name, **kwargs)
    self.nonum = nonum


class RsvqaLr(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for RSVQA-LR dataset."""

  VERSION = tfds.core.Version('1.0.2')
  RELEASE_NOTES = {
      '1.0.0': 'First release.',
      '1.0.1': 'Rename binned values.',
      '1.0.2': 'Removed explicit png image encoding.',
      }

  BUILDER_CONFIGS = [
      RsvqaLrConfig(nonum=False),
      RsvqaLrConfig(nonum=True),
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
        for split in ('train', 'val', 'test')
    }

  def _generate_examples(self, split):
    """Yields (key, example) tuples."""
    questions_path = os.path.join(PATH + f'LR_split_{split}_questions.json')
    answers_path = os.path.join(PATH + f'LR_split_{split}_answers.json')
    images_path = os.path.join(PATH + 'Images_LR')

    with open(questions_path, 'r') as f:
      questions = json.loads(f.read())['questions']
    with open(answers_path, 'r') as f:
      answers = json.loads(f.read())['answers']

    for q, a in zip(questions, answers):
      assert q['active'] == a['active']
      if not q['active']:
        continue
      if self.builder_config.nonum and q['type'] == 'count':
        continue
      assert q['answers_ids'] == [a['id']]
      assert q['id'] == a['question_id']

      filename = f'{q["img_id"]}.tif'
      img = read_tif(os.path.join(images_path, filename))
      yield q['id'], {
          'question_id': q['id'],
          'filename': filename,
          'image': img,
          'question': q['question'],
          'question_type': q['type'],
          'answers': [bin_answer(a['answer'], q['type'])],
          'raw_answers': [a['answer']],
      }


def bin_answer(answer, question_type):
  """Bins answers into expected ranges."""
  if question_type == 'count':
    count = int(answer)
    if count == 0:
      return '0'
    elif count <= 10:
      return 'between 1 and 10'
    elif count <= 100:
      return 'between 11 and 100'
    elif count <= 1000:
      return 'between 101 and 1000'
    else:
      return 'more than 1000'
  return answer


def read_tif(path):
  with open(path, 'rb') as f:
    img = tfds.core.lazy_imports.tifffile.imread(io.BytesIO(f.read()))
  return img.astype(np.uint8)
