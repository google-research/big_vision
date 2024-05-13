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
r"""Implements CharQA in TFDS structure.

It's small data, so simple to run locally. First, copy the data to local disk:

  mkdir -p /tmp/data
  wget -O /tmp/data/chartqa.zip https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip?download=true
  unzip /tmp/data/chartqa.zip

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=chartqa

Example to load:

  import tensorflow_datasets as tfds
  dataset_augmented = tfds.load('chartqa/augmented', split='train', data_dir='/tmp/tfds')
"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """ChartQA dataset."""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{masry-etal-2022-chartqa,
    title = "{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning",
    author = "Masry, Ahmed  and
      Do, Xuan Long  and
      Tan, Jia Qing  and
      Joty, Shafiq  and
      Hoque, Enamul",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.177",
    doi = "10.18653/v1/2022.findings-acl.177",
    pages = "2263--2279",
    abstract = "Charts are very popular for analyzing data. When exploring charts, people often ask a variety of complex reasoning questions that involve several logical and arithmetic operations. They also commonly refer to visual features of a chart in their questions. However, most existing datasets do not focus on such complex reasoning questions as their questions are template-based and answers come from a fixed-vocabulary. In this work, we present a large-scale benchmark covering 9.6K human-written questions as well as 23.1K questions generated from human-written chart summaries. To address the unique challenges in our benchmark involving visual and logical reasoning over charts, we present two transformer-based models that combine visual features and the data table of the chart in a unified way to answer questions. While our models achieve the state-of-the-art results on the previous datasets as well as on our benchmark, the evaluation also reveals several challenges in answering complex reasoning questions.",
}
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_CHARTQA_PATH = '/tmp/data/ChartQA Dataset/'


class ChartQAConfig(tfds.core.BuilderConfig):
  """Configuration to build the dataset."""
  pass


class ChartQA(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ChartQA dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}
  BUILDER_CONFIGS = [
      ChartQAConfig(name='human', description='Human set'),
      ChartQAConfig(name='augmented', description='Augmented set'),
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
        }),
        homepage='https://github.com/vis-nlp/ChartQA',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split, self.builder_config.name)
            for split in ('val', 'train', 'test')}

  def _generate_examples(self, split: str, source: str):
    """Yields (key, example) tuples from test set."""
    annot_fname = os.path.join(_CHARTQA_PATH, split, f'{split}_{source}.json')

    with open(annot_fname, 'r') as f:
      data = json.loads(f.read())

    for idx, v in enumerate(data):
      yield idx, {
          'question_id': idx,
          'image/filename': v['imgname'],
          'image': os.path.join(_CHARTQA_PATH, split, 'png', v['imgname']),
          'question': v['query'],
          'answer': v['label'],
      }
