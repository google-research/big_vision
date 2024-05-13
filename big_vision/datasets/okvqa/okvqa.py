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
r"""Implements the OKVQA dataset for TFDS.

Download the required files from https://okvqa.allenai.org/download.html:

mkdir -p /tmp/tfds
cd /tmp/tfds/
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip
unzip val2014.zip
unzip train2014.zip
unzip OpenEnded_mscoco_train2014_questions.json.zip
unzip OpenEnded_mscoco_val2014_questions.json.zip
unzip mscoco_train2014_annotations.json.zip
unzip mscoco_val2014_annotations.json.zip

Then, run conversion locally (make sure to install tensorflow-datasets for the
`tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=okvqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('okvqa', split='val', data_dir='/tmp/tfds')
"""

import json
import os
from typing import Any
import numpy as np
import tensorflow_datasets as tfds

_DESCRIPTION = """
OKVQA addresses the task of VQA with outside knowledge.
This version of the dataset contains:
- Questions + Answers from OKVQA.
- Images from COCO.
"""

_CITATION = """
@InProceedings{okvqa,
author = {Kenneth Marino and Mohammad Rastegari and Ali Farhadi and Roozbeh Mottaghi},
title = {OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019},
}
"""

ANNOTATION_FILE = {
    'train': 'mscoco_train2014_annotations.json',
    'val': 'mscoco_val2014_annotations.json',
}
QUESTIONS_FILE = {
    'train': 'OpenEnded_mscoco_train2014_questions.json',
    'val': 'OpenEnded_mscoco_val2014_questions.json',
}
QUESTION_TYPES = {
    'one': 'Vehicles and Transportation',
    'two': 'Brands, Companies and Products',
    'three': 'Objects, Material and Clothing',
    'four': 'Sports and Recreation',
    'five': 'Cooking and Food',
    'six': 'Geography, History, Language and Culture',
    'seven': 'People and Everyday life',
    'eight': 'Plants and Animals',
    'nine': 'Science and Technology',
    'ten': 'Weather and Climate',
    'other': 'Other',
}


# When running locally (recommended), copy files as above an use these:
_OKVQA_PATH = '/media/scratch/okvqa'


class OkVqa(tfds.core.GeneratorBasedBuilder):
  """Import COCO dataset for OKVQA with KAT features."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Changed to array record format.'}
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  In manual_dir/ you should have a directory okvqa which contains the
  following files and directories:
  From the OKVQA dataset:
  - mscoco_train2014_annotations.json
  - mscoco_val2014_annotations.json
  - OpenEnded_mscoco_train2014_questions.json
  - OpenEnded_mscoco_val2014_questions.json
  - train2014.zip
  - val2014.zip
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(None, None, 3)),
        'image_id': tfds.features.Scalar(dtype=np.int64),
        'answer_type': tfds.features.Text(),
        'answers': tfds.features.Sequence(tfds.features.Text()),
        'answers_confidence': tfds.features.Tensor(shape=[10], dtype=np.bool_),
        'answers_raw': tfds.features.Sequence(tfds.features.Text()),
        'question_id': tfds.features.Scalar(dtype=np.int64),
        'question_type': tfds.features.Text(),
        'question_type_readable': tfds.features.Text(),
        'question': tfds.features.Text(),
    })

    return tfds.core.DatasetInfo(
        builder=self,
        features=features,
        description=_DESCRIPTION,
        supervised_keys=None,
        homepage='https://okvqa.allenai.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager) -> ...:
    """Call the function which defines the splits."""
    # data_dir = dl_manager.manual_dir
    data_dir = _OKVQA_PATH
    return {
        'train': self._generate_examples(data_dir, 'train'),
        'val': self._generate_examples(data_dir, 'val'),
    }

  def _generate_examples(self, data_dir: str, split: str) -> ...:
    annotations = get_okvqa_annotations(data_dir, split)

    for question_id, annotation in annotations.items():
      image_id = annotation['image_id']

      # Sanity check.
      if len(annotation['answers']) != 10:
        num_answers = len(annotation['answers'])
        raise ValueError(
            f'The number of answers for {image_id} is not 10 but {num_answers}')

      feature_dict = {
          'image': self.get_image_path(data_dir, split, image_id),
          'image_id': image_id,
          'answer_type': annotation['answer_type'],
          'answers': [a['answer'] for a in annotation['answers']],
          'answers_confidence': _get_answer_confidence(annotation['answers']),
          'answers_raw': [a['raw_answer'] for a in annotation['answers']],
          'question_id': annotation['question_id'],
          'question_type': annotation['question_type'],
          'question_type_readable': QUESTION_TYPES[annotation['question_type']],
          'question': annotation['question'],
      }
      yield f'{question_id}', feature_dict

  def get_image_path(self, data_dir: str, split: str, image_id: int) -> np.ndarray:
    subdir = {'train': 'train2014', 'val': 'val2014'}[split]
    return f'{data_dir}/{subdir}/COCO_{subdir}_{image_id:012d}.jpg'


def _get_answer_confidence(answers: list[dict[str, str]]) -> np.ndarray:
  """Get OKVQA answer confidences as bool."""
  confidences = []
  for a in answers:
    confidence = a['answer_confidence']
    if confidence == 'yes':
      confidences.append(True)
    elif confidence == 'no':
      confidences.append(False)
    else:
      raise ValueError(f'Unknown confidence: {confidence}')
  return np.array(confidences, dtype=bool)


def _read_json(
    data_dir: str, file: str, key: str
) -> dict[int, dict[str, Any]]:
  with open(os.path.join(data_dir, file)) as f:
    data = json.load(f)
  questions = {d['question_id']: d for d in data[key]}
  return questions


def get_okvqa_annotations(
    data_dir: str, split: str
) -> dict[int, dict[str, Any]]:
  """Return okvqa annotations (quesions and answers) as dictionary."""
  questions = _read_json(data_dir, QUESTIONS_FILE[split], 'questions')
  annotations = _read_json(data_dir, ANNOTATION_FILE[split], 'annotations')

  assert len(annotations) == len(questions)
  for question_id, question in questions.items():
    assert question['image_id'] == annotations[question_id]['image_id']
    assert question['question_id'] == annotations[question_id]['question_id']
    annotations[question_id]['question'] = question['question']

  return annotations
