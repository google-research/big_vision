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
r"""Import VQAv2 into TFDS format. Uses coco-2014 images.

It's small data, so simple to run locally. First, download all the data:

  mkdir /tmp/data/ ; cd /tmp/data
  wget http://images.cocodataset.org/zips/{train2014,val2014,test2015}.zip
  wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_{Train,Val,Test}_mscoco.zip
  wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_{Train,Val}_mscoco.zip
  unzip '*.zip'

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=vqa

It runs at around 750 examples/sec, so takes around 25min for the 1.2M questions.
Each question is an example; images are repeated, a bit wasteful, but disk is cheap.


Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('vqa', split='train', data_dir='/tmp/tfds')
"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_VQAV2_PATH = '/tmp/data'
_IMAGE_PATH = '/tmp/data'


_CITATION = (
    '@InProceedings{balanced_vqa_v2,'
    'author = {Yash Goyal and Tejas Khot and '
    'Douglas Summers{-}Stay and Dhruv Batra and Devi Parikh},'
    'title = {Making the {V} in {VQA} Matter: Elevating the Role of Image'
    'Understanding in {V}isual {Q}uestion {A}nswering},'
    'booktitle = {Computer Vision and Pattern Recognition (CVPR)},'
    'year = {2017},}')


class Vqa(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for VQAv2 dataset."""

  VERSION = tfds.core.Version('3.0.0')
  RELEASE_NOTES = {'3.0.0': 'Format as needed for OpenPaLI'}

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description='The VQAv2 dataset.',
        features=tfds.features.FeaturesDict({
            'image/id': np.int32,
            'image/filename': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'question_id': np.int32,
            'question_type': tfds.features.Text(),
            'question_text': tfds.features.Text(),
            'answer_type': tfds.features.Text(),
            'answers': tfds.features.Sequence(tfds.features.Text()),
            'answer_confidences': tfds.features.Sequence(
                tfds.features.ClassLabel(names=['no', 'maybe', 'yes'])),
            'top_answer': tfds.features.Text(),
        }),
        homepage='https://visualqa.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
        'train': self._generate_examples('train2014'),
        'validation': self._generate_examples('val2014'),
        'test': self._generate_examples('test2015'),
        'test-dev': self._generate_examples('test-dev2015', 'test2015'),
    }

  def _generate_examples(self, split, image_folder=None):
    """Yields (key, example) tuples from test set."""
    image_folder = image_folder or split

    # The questions file has fields image_id, question, question_id.
    with open(os.path.join(
        _VQAV2_PATH, f'v2_OpenEnded_mscoco_{split}_questions.json')) as f:
      examples = json.load(f)['questions']

    # The questions file has fields: image_id, question_id, answers,
    # answer_type, question_type, multiple_choice_answer.
    if 'test' not in split:
      with open(os.path.join(
          _VQAV2_PATH, f'v2_mscoco_{split}_annotations.json')) as f:
        annots = {a['question_id']: a for a in json.load(f)['annotations']}

    for ex in examples:
      qid = ex['question_id']
      ex = {
          'image/id': ex['image_id'],
          'question_id': qid,
          'question_text': ex['question'],
      }
      if 'test' not in split:
        fname = f'COCO_{image_folder}_{ex["image/id"]:012d}.jpg'
        ex['image/filename'] = fname
        ex['image'] = os.path.join(_IMAGE_PATH, image_folder, fname)
        ann = annots[qid]
        ex['question_type'] = ann['question_type']
        ex['answer_type'] = ann['answer_type']
        ex['answers'] = [a['answer'] for a in ann['answers']]
        ex['answer_confidences'] = [a['answer_confidence']
                                    for a in ann['answers']]
        ex['top_answer'] = ann['multiple_choice_answer']
      else:
        # For test images, a few are from the wrong year...
        fname = f'COCO_{image_folder}_{ex["image/id"]:012d}.jpg'
        ex['image/filename'] = fname
        if os.path.isfile(path := os.path.join(_IMAGE_PATH, image_folder, fname)):
          ex['image'] = path
        else:
          print(ex['image/id'])
          continue
        ex['question_type'] = ''
        ex['answer_type'] = ''
        ex['answers'] = []
        ex['answer_confidences'] = []
        ex['top_answer'] = ''
      yield qid, ex
