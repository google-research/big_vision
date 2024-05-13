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
r"""Generates GQA in a TFDS-ready structure, using Beam.

Instructions below are to generate the dataset with a *local* Beam pipeline.
It's advisable to run the Beam job on Google Cloud Dataflow, see
  https://www.tensorflow.org/datasets/beam_datasets.
for more details, which would significantly speed up generation. This would
involve uploading the locally downloaded data to a GCS bucket, and then
adding in the Beam pipeline options and your GCP/GCS bucket details
to the `tfds build` command below (as detailed in the link).

First, copy the data to local disk:

  mkdir -p /tmp/data/gqa
  wget -O /tmp/data/gqa/question1.2.zip https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip?download=true
  unzip /tmp/data/gqa/question1.2.zip
  mv /tmp/data/gqa/question1.2/* /tmp/data/gqa/
  wget -O /tmp/data/gqa/images.zip https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip?download=true
  unzip /tmp/data/gqa/images.zip

Then, run conversion (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=gqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('gqa', split='testdev_balanced', data_dir='/tmp/tfds')

Some statistics:
  train_all: 14305356 examples
  train_balanced: 943000 examples
  val_all: 2011853 examples
  val_balanced: 132062 examples
  testdev_all: 172174 examples
  testdev_balanced: 12578 examples
"""
import glob
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """GQA: Visual Reasoning in the Real World."""

# pylint: disable=line-too-long
_CITATION = """
@article{DBLP:journals/corr/abs-2306-14610,
  author       = {Drew Hudson and
                  Christopher Manning},
  title        = {GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering},
  journal      = {CVPR},
  volume       = {abs/1902.09506},
  year         = {2019},
  url          = {https://doi.org/10.48550/arXiv.1902.09506},
  doi          = {10.48550/arXiv.1902.09506},
  eprinttype    = {arXiv},
  eprint       = {1902.09506},
  timestamp    = {Tue, 25 Jun 2019 00:00:00 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1902-09506},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""
# pylint: enable=line-too-long


_DATA_PATH = '/tmp/data/gqa/'


class GQA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for GQA dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'example_id': tfds.features.Scalar(np.int64),
            'image/id': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'question': tfds.features.Text(),
            'answer': tfds.features.Text(),
            'full_answer': tfds.features.Text(),
            'is_balanced': tfds.features.Scalar(np.bool_),
        }),
        homepage='https://cs.stanford.edu/people/dorarad/gqa/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    splits = [
        # 'debug',
        'train_all',
        'train_balanced',
        'testdev_all',
        'testdev_balanced',
        'val_all',
        'val_balanced',
        'challenge_all',
        'challenge_balanced',
    ]
    return {split: self._generate_examples(split) for split in splits}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples from dataset."""
    if split == 'train_all':
      train_json_dir = os.path.join(_DATA_PATH, 'train_all_questions', '*.json')
      json_files = glob.glob(train_json_dir)
    else:
      json_files = [os.path.join(_DATA_PATH, f'{split}_questions.json')]

    def _prepare_data(json_path):
      with open(os.path.join(json_path)) as f:
        annotations = json.load(f)
      return [(k, v) for k, v in annotations.items()]

    def _process_example(entry):
      question_id, question_data = entry
      image_id = question_data['imageId']
      image_path = os.path.join(_DATA_PATH, 'images', f'{image_id}.jpg')
      answer = question_data['answer'] if 'answer' in question_data else ''
      if 'fullAnswer' in question_data:
        full_answer = question_data['fullAnswer']
      else:
        full_answer = ''

      example = {
          'example_id': question_id,
          'image/id': image_id,
          'image': image_path,
          'question': question_data['question'],
          'answer': answer,
          'full_answer': full_answer,
          'is_balanced': question_data['isBalanced'],
      }
      return question_id, example

    beam = tfds.core.lazy_imports.apache_beam
    return (
        beam.Create(json_files)
        | beam.FlatMap(_prepare_data)
        | beam.Reshuffle()
        | beam.Map(_process_example)
    )
