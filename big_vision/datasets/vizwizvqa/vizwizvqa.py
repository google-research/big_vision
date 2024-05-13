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
r"""Implements VizWizVQA dataset in TFDS structure.

It's small data, so simple to run locally. First, copy the data to local disk:

  mkdir -p /tmp/data/vizwizvqa

  wget -O  https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip /tmp/data/vizwizvqa
  wget -O  https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip /tmp/data/vizwizvqa
  wget -O https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip /tmp/data/vizwizvqa

Then, run conversion locally
(make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=vizwizvqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('vizwizvqa', split='train', data_dir='/tmp/tfds')
"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """VizWiz VQA Dataset."""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{gurari2018vizwiz,
  title={Vizwiz grand challenge: Answering visual questions from blind people},
  author={Gurari, Danna and Li, Qing and Stangl, Abigale J and Guo, Anhong and Lin, Chi and Grauman, Kristen and Luo, Jiebo and Bigham, Jeffrey P},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3608--3617},
  year={2018}
}
}
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_VIZWIZVQA_PATH = '/tmp/data/vizwizvqa/'


class VizWizVQA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for VizWizVQA dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}

  def _info(self):
    """Returns the metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'question': tfds.features.Text(),
            'image/filename': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'answers': tfds.features.Sequence(tfds.features.Text()),
            # can be "yes" "no" and "maybe" strings
            'answer_confidences': tfds.features.Sequence(tfds.features.Text()),
            'answerable': tfds.features.Scalar(np.int32),
            'question_id': tfds.features.Scalar(np.int32),
        }),
        supervised_keys=None,
        homepage='https://vizwiz.org/tasks-and-datasets/vqa/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split)
            for split in ('val', 'train', 'test',)}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples from test set."""
    annot_fname = os.path.join(_VIZWIZVQA_PATH, 'annotations', f'{split}.json')

    with open(annot_fname, 'r') as f:
      data = json.loads(f.read())

    for v in data:

      answers = []
      answer_confidences = []

      image_file = v['image']
      answerable = -1
      if split != 'test':
        for answer in v['answers']:
          # A couple of answers in the train set are empty strings.
          if not answer['answer']:
            continue
          answers.append(answer['answer'])
          answer_confidences.append(answer['answer_confidence'])
        answerable = v['answerable']

      question_id = image_file[:-4]
      question_id = int(question_id.split('_')[-1])

      yield v['image'], {
          'question': v['question'],
          'image/filename': image_file,
          'question_id': question_id,
          'image': os.path.join(_VIZWIZVQA_PATH, split, image_file),
          'answers': answers,
          'answer_confidences': answer_confidences,
          'answerable': answerable,
      }
