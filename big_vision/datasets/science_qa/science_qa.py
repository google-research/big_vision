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
r"""Implements ScienceQA train/val/test-set in TFDS structure.

First, download the science QA dataset from their website https://scienceqa.github.io/#download
  - mkdir -p /tmp/data/ScienceQA_DATA
  - From Google Drive: https://drive.google.com/corp/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev
Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):
  - cd big_vision/datasets
  - env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=science_qa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load(
      'science_qa', split='train',
      data_dir='/tmp/tfds')

"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """Sci QA test-set."""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{lu2022learn,
    title={Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering},
    author={Lu, Pan and Mishra, Swaroop and Xia, Tony and Qiu, Liang and Chang, Kai-Wei and Zhu, Song-Chun and Tafjord, Oyvind and Clark, Peter and Ashwin Kalyan},
    booktitle={The 36th Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_SCIQA_PATH = '/tmp/data/ScienceQA_DATA/'
# _IMAGE_COCO_PATH = '/tmp/data/val2014'

_ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class ScienceQA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ScienceQA dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'question': tfds.features.Text(),
            'choices': tfds.features.Sequence(tfds.features.Text()),
            'answer': tfds.features.Scalar(np.int32),
            'hint': tfds.features.Text(),
            'task': tfds.features.Text(),
            'grade': tfds.features.Text(),
            'subject': tfds.features.Text(),
            'topic': tfds.features.Text(),
            'category': tfds.features.Text(),
            'skill': tfds.features.Text(),
            'lecture': tfds.features.Text(),
            'solution': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='png'),
            'indexed_choices': tfds.features.Text(),
            'indexed_answer': tfds.features.Text(),
        }),
        supervised_keys=None,
        homepage='https://github.com/lupantech/ScienceQA/tree/main',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
        split: self._generate_examples(split)
        for split in ('train', 'test', 'val')
    }

  def _generate_examples(self, split):
    """Yields (key, example) tuples from test set."""
    annot_fname = os.path.join(_SCIQA_PATH, 'problems.json')

    with open(annot_fname, 'r') as f:
      data = json.loads(f.read())

    for k, v in data.items():
      if v['split'] == split:  #  "split":"train"
        image = v['image']
        # Science QA contains the example without image as well. As this
        # conversion is for VQA tasks, we dropped the examples without Image.
        # TODO: Include the examples without image, and udpate the
        # downstream pipeline to skip the examples without image, instead of
        # doing it at pre-processing.
        if image:
          image = os.path.join(f'{_SCIQA_PATH}/{split}/{k}/', f'{image}')
        else:
          # image = None
          continue
        question = v['question']
        choices = v['choices']
        answer = v['answer']
        hint = v['hint']
        if not hint:
          hint = 'N/A'  # align with orignal github implementation
        task = v['task']
        grade = v['grade']
        subject = v['subject']
        topic = v['topic']
        category = v['category']
        skill = v['skill']
        lecture = v['lecture']
        solution = v['solution']
        split = v['split']
        indexed_choices = ', '.join(
            f'({_ALPHABETS[i]}) {c}' for i, c in enumerate(choices)
        )
        indexed_answer = _ALPHABETS[int(answer)]
        yield int(k), {
            'question': question,
            'choices': choices,
            'answer': answer,
            'hint': hint,
            'task': task,
            'grade': grade,
            'subject': subject,
            'topic': topic,
            'category': category,
            'skill': skill,
            'lecture': lecture,
            'solution': solution,
            'image': image,
            'indexed_choices': indexed_choices,
            'indexed_answer': indexed_answer,
        }
