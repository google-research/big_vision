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
r"""Implements POPE test-set in TFDS structure.

It's small data, so simple to run locally. First, copy the data to local disk:
First download json files from https://github.com/AoiDragon/POPE; then download
MSCOCO (val 2014) images from https://cocodataset.org/#download

  mkdir -p /tmp/data/pope/
  mkdir -p /tmp/data/pope/pope/
  mkdir -p /tmp/data/pope/images/
  git clone https://github.com/AoiDragon/POPE.git
  cp POPE/output/coco/* /tmp/data/pope/pope/
  wget http://images.cocodataset.org/zips/val2014.zip
  unzip val2014.zip
  cp -r val2014/ /tmp/data/pope/images/

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=pope

Example to load:

  import tensorflow_datasets as tfds
  dataset_random = tfds.load('pope/pope_random', split='test', data_dir='/tmp/tfds')
  dataset_popular = tfds.load('pope/pope_popular', split='test', data_dir='/tmp/tfds')
  dataset_adversarial = tfds.load('pope/pope_adversarial', split='test', data_dir='/tmp/tfds')

"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """POPE dataset."""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{li-etal-2023-evaluating,
    title = "Evaluating Object Hallucination in Large Vision-Language Models",
    author = "Li, Yifan  and
      Du, Yifan  and
      Zhou, Kun  and
      Wang, Jinpeng  and
      Zhao, Xin  and
      Wen, Ji-Rong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.20",
    doi = "10.18653/v1/2023.emnlp-main.20",
    pages = "292--305",
    abstract = "Inspired by the superior language abilities of large language models (LLM), large vision-language models (LVLM) have been recently proposed by integrating powerful LLMs for improving the performance on complex multimodal tasks. Despite the promising progress on LVLMs, we find that they suffer from object hallucinations, i.e., they tend to generate objects inconsistent with the target images in the descriptions. To investigate it, this work presents the first systematic study on object hallucination of LVLMs. We conduct the evaluation experiments on several representative LVLMs, and show that they mostly suffer from severe object hallucination issues. We further discuss that the visual instructions may influence the hallucination, and find that: objects that frequently appear in the visual instructions or co-occur with the image objects are obviously prone to be hallucinated by LVLMs. Besides, we further design a polling-based query method called POPE for better evaluation of object hallucination. Experiment results show that our POPE can evaluate object hallucination in a more stable and flexible way.",
}
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above and use these:
_POPE_PATH = '/tmp/data/pope/'


class POPEConfig(tfds.core.BuilderConfig):
  """Configuration to build the dataset."""

  pass


class POPE(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for POPE dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}
  BUILDER_CONFIGS = [
      POPEConfig(name='pope_random', description='Random set'),
      POPEConfig(name='pope_popular', description='Popular set'),
      POPEConfig(name='pope_adversarial', description='Adversarial set'),
  ]

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
            'answer': tfds.features.Text(),
            'thing': tfds.features.Text(),
        }),
        supervised_keys=None,
        homepage='https://github.com/AoiDragon/POPE',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {'test': self._generate_examples('test', self.builder_config.name)}

  def _generate_examples(self, split: str, source: str):
    """Yields (key, example) tuples from test set."""
    annot_fname = os.path.join(
        _POPE_PATH, f'pope/coco_{source}.json'
    )

    with open(annot_fname, 'r') as f:
      data = [json.loads(line) for line in f]

    for idx, v in enumerate(data):
      question = v['text']
      thing = (
          question.replace('Is there an ', '')
          .replace('Is there a ', '')
          .replace(' in the image?', '')
      )
      yield idx, {
          'question_id': idx,
          'image/filename': v['image'],
          'image': os.path.join(_POPE_PATH, 'images/val2014/', v['image']),
          'question': question,
          'answer': v['label'],
          'thing': thing,
      }
