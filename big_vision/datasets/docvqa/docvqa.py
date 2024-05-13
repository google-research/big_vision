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
r"""Implements DocVQA in TFDS structure.

It's small data, so simple to run locally. First, copy the data to local disk.
An account will be needed in https://rrc.cvc.uab.es/?ch=17&com=downloads and
from there the task annotations and images can be fetched separatedly.

  mkdir -p /tmp/data/docvqa
  <COPY AND DECOMPRESS DOWNLOADED FILES HERE>

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=docvqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('docvqa', split='val', data_dir='/tmp/tfds')
"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """DocVQA dataset."""

# pylint: disable=line-too-long
_CITATION = """
@article{DBLP:journals/corr/abs-2007-00398,
  author       = {Minesh Mathew and
                  Dimosthenis Karatzas and
                  R. Manmatha and
                  C. V. Jawahar},
  title        = {DocVQA: {A} Dataset for {VQA} on Document Images},
  journal      = {CoRR},
  volume       = {abs/2007.00398},
  year         = {2020},
  url          = {https://arxiv.org/abs/2007.00398},
  eprinttype    = {arXiv},
  eprint       = {2007.00398},
  timestamp    = {Mon, 06 Jul 2020 15:26:01 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2007-00398.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_DOCVQA_PATH = '/tmp/data/docvqa/'


class DocVQA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for DocVQA dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}

  def _info(self):
    """Returns the metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'question_id': tfds.features.Scalar(np.int32),
            'image/filename': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='png'),
            'question': tfds.features.Text(),
            'answers': tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,
        homepage='https://www.docvqa.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split)
            for split in ('val', 'train', 'test')}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples from split."""
    suffix = '' if split == 'test' else '_withQT'
    with open(os.path.join(_DOCVQA_PATH, f'{split}_v1.0{suffix}.json')) as f:
      data = json.load(f)
    for v in data['data']:
      question_id = v['questionId']
      yield question_id, {
          'question_id': question_id,
          'image/filename': v['image'],
          'image': os.path.join(_DOCVQA_PATH, split, v['image']),
          'question': v['question'],
          'answers': v.get('answers', []),
      }
