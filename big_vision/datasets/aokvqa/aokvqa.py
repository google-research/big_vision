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

Download the required files from https://aokvqa.allenai.org/download.html:

mkdir -p /tmp/tfds
cd /tmp/tfds/
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz
unzip val2017.zip
unzip train2017.zip
unzip test2017.zip
tar xzf aokvqa_v1p0.tar.gz

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=aokvqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('aokvqa', split='val', data_dir='/tmp/tfds')
"""

import json
import os
from typing import Any
import numpy as np
import tensorflow_datasets as tfds

_DESCRIPTION = """
A-OKVQA addresses the task of VQA with outside knowledge.
It is a follow-up dataset of OKVQA.

This version of the dataset contains:
- Questions + Answers + Multiple Choice Answers + Rationales from A-OKVQA.
- Images from COCO.
"""

_CITATION = """
@article{AOKVQA,
  title={A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge},
  author={Dustin Schwenk and Apoorv Khandelwal and Christopher Clark and Kenneth Marino and Roozbeh Mottaghi},
  journal={arXiv},
  year={2022},
}
"""

ANNOTATION_FILES = {
    'train': 'aokvqa_v1p0_train.json',
    'val': 'aokvqa_v1p0_val.json',
    'test': 'aokvqa_v1p0_test.json',
}


# When running locally (recommended), copy files as above an use these:
_AOKVQA_PATH = '/tmp/tfds'


class AOkVqa(tfds.core.GeneratorBasedBuilder):
  """AOKVQA dataset for TFDS."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'ArrayRecord version.'}
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  In manual_dir/ you should have a directory a_ok_vqa which contains the
  following files and directories:
  From the A-OKVQA dataset:
  - aokvqa_v1p0_train.json
  - aokvqa_v1p0_val.json
  - aokvqa_v1p0_test.json
  It also requires the COCO data files.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(None, None, 3)),
        'image_id': tfds.features.Scalar(dtype=np.int64),
        'direct_answers': tfds.features.Sequence(tfds.features.Text()),
        'direct_answer_is_difficult': tfds.features.Scalar(dtype=np.bool_),
        'multiple_choice_possible_answers':  # List of 4 possible answers.
            tfds.features.Sequence(tfds.features.Text()),
        'multiple_choice_correct_idx':  # Integer from 0-3.
            tfds.features.Scalar(dtype=np.int32),
        'answer_rationales': tfds.features.Sequence(tfds.features.Text()),
        'question': tfds.features.Text(),
        'question_id': tfds.features.Text(),
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
    data_dir = _AOKVQA_PATH
    return {
        'train': self._generate_examples(data_dir, 'train'),
        'val': self._generate_examples(data_dir, 'val'),
        'test': self._generate_examples(data_dir, 'test'),
    }

  def _generate_examples(self, data_dir: str, split: str) -> ...:
    annotations = get_annotations(data_dir, split)

    for question_id, feature_dict in annotations.items():
      image_id = feature_dict['image_id']

      # Add image and GT segmentatio labels from total_transfer.
      feature_dict['image'] = self.get_image_path(data_dir, split, image_id)

      # Add dummy features for several features in the test set.
      if split not in ['train', 'val']:
        assert split == 'test', f'Unknown split: {split}'
        feature_dict['multiple_choice_correct_idx'] = -1
        feature_dict['direct_answers'] = []
        feature_dict['answer_rationales'] = []
      yield f'{question_id}', feature_dict

  def get_image_path(self, data_dir: str, split: str, image_id: int) -> np.ndarray:
    return f'{data_dir}/{split}2017/{image_id:012d}.jpg'


def get_annotations(
    data_dir: str, split: str) -> dict[int, dict[str, Any]]:
  """Return okvqa annotations (quesions and answers) as dictionary."""
  path = os.path.join(data_dir, ANNOTATION_FILES[split])
  with open(path) as f:
    annotations = json.load(f)

  aokvqa_annotations = {}
  for annotation in annotations:
    # Sanity checks
    assert len(annotation['choices']) == 4

    question_id = annotation['question_id']

    aokvqa_annotations[question_id] = {
        'image_id': annotation['image_id'],
        'direct_answer_is_difficult': annotation['difficult_direct_answer'],
        'multiple_choice_possible_answers': annotation['choices'],
        'question': annotation['question'],
        'question_id': annotation['question_id'],
    }

    # Get answers and rationales for train and val only, not for test.
    if split in ['train', 'val']:
      assert len(annotation['direct_answers']) == 10
      assert len(annotation['rationales']) == 3

      aokvqa_annotations[question_id]['direct_answers'] = annotation[
          'direct_answers']
      aokvqa_annotations[question_id]['answer_rationales'] = annotation[
          'rationales']
      aokvqa_annotations[question_id]['multiple_choice_correct_idx'] = (
          annotation['correct_choice_idx'])

  return aokvqa_annotations
