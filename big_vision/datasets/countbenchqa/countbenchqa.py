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
"""Import CountBenchQA dataset (CountBench dataset with added QA annotations).

It's small data, so simple to run locally. First, download all the data:

  mkdir /tmp/data/ ; cd /tmp/data
  wget https://huggingface.co/datasets/nielsr/countbench/resolve/main/data/train-00000-of-00001-cf54c241ba947306.parquet
  wget https://raw.githubusercontent.com/teaching-clip-to-count/teaching-clip-to-count.github.io/main/CountBench.json

Then, update the PATHs below and run conversion locally like so:

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=countbenchqa

The dataset contains 540 images so the dataset creation is very quick.

There is a single split called huggingface to denote that the images come from
the hugginface parquet file.
"""

import io
import json

import numpy as np
import pandas as pd
import PIL
import tensorflow_datasets as tfds


# Huggingface dataset path; this is missing about 10% of the images.
_COUNTBENCH_PARQUET_PATH = '/tmp/data/train-00000-of-00001-cf54c241ba947306.parquet'
# Public path to the original CountBench JSON file.
_COUNTBENCH_JSON_PATH = '/tmp/data/CountBench.json'
# VQA annotations
_QA_JSON_PATH = 'countbenchqa/data/countbench_paired_questions.json'

_DESCRIPTION = """
CountBench: We introduce a new object counting benchmark called CountBench,
  automatically curated (and manually verified) from the publicly available
  LAION-400M image-text dataset. CountBench contains a total of 540 images
  containing between two and ten instances of a particular object, where their
  corresponding captions reflect this number.

CountBenchQA: Each image is paired with a manually generated question about the
  number of objects in the image to turn CountBench into a VQA task.
"""

_CITATION = """
@article{beyer2024paligemma,
      title={{PaliGemma: A versatile 3B VLM for transfer}},
      author={Lucas Beyer and Andreas Steiner and André Susano Pinto and Alexander Kolesnikov and Xiao Wang and Daniel Salz and Maxim Neumann and Ibrahim Alabdulmohsin and Michael Tschannen and Emanuele Bugliarello and Thomas Unterthiner and Daniel Keysers and Skanda Koppula and Fangyu Liu and Adam Grycner and Alexey Gritsenko and Neil Houlsby and Manoj Kumar and Keran Rong and Julian Eisenschlos and Rishabh Kabra and Matthias Bauer and Matko Bošnjak and Xi Chen and Matthias Minderer and Paul Voigtlaender and Ioana Bica and Ivana Balazevic and Joan Puigcerver and Pinelopi Papalampidi and Olivier Henaff and Xi Xiong and Radu Soricut and Jeremiah Harmsen and Xiaohua Zhai},
      year={2024},
      journal={arXiv preprint arXiv:2407.07726}
}

@article{paiss2023countclip,
      title={{Teaching CLIP to Count to Ten}},
      author={Paiss, Roni and Ephrat, Ariel and Tov, Omer and Zada, Shiran and Mosseri, Inbar and Irani, Michal and Dekel, Tali},
      year={2023},
      journal={arXiv preprint arXiv:2302.12066}
}
"""

_HOMEPAGE = 'https://teaching-clip-to-count.github.io/'


class CountbenchQA(tfds.core.GeneratorBasedBuilder):
  """Create CountbenchQA dataset."""

  VERSION = tfds.core.Version('1.2.0')
  RELEASE_NOTES = {'1.1.0': 'Add `huggingface` split.',
                   '1.2.0': 'Fix image loading for `huggingface` split.'}
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  There are two parts which should be downloaded:
  * Countbench from Huggingface
  * Questions found in `data/countbench_paired_questions.json`
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(None, None, 3)),
        'image_id': tfds.features.Scalar(dtype=np.int32),
        'question': tfds.features.Text(),
        'text': tfds.features.Text(),
        'image_url': tfds.features.Text(),
        'number': tfds.features.Scalar(dtype=np.int32),
    })

    return tfds.core.DatasetInfo(
        builder=self,
        features=features,
        description=_DESCRIPTION,
        supervised_keys=None,
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Call the function which defines the splits."""
    del dl_manager
    return {
        'huggingface': self._generate_examples(split='huggingface'),
    }

  def _generate_examples_hf(self):
    """Generate examples from Huggingface parquet file.

    Note that the parquet file provided on Huggingface is missing about 10%
    of the images as can be verified by running
    ```
        import pyarrow.parquet as pq
        with open(_COUNTBENCH_PARQUET_PATH, 'rb') as f:
          x = pq.read_table(f)
        sum([x['image'][i].is_valid for i in range(len(x['image']))])  # result: 491
    ```

    Yields:
      An index and a dictionary with features.
    """
    with open(_COUNTBENCH_PARQUET_PATH, 'rb') as f:
      df = pd.read_parquet(f)

    with open(_QA_JSON_PATH, 'r') as fq:
      df_question = pd.read_json(fq)

    df['question'] = df_question

    for idx, row in df.iterrows():
      # Some entries have no image.
      if row['image'] is None:
        continue
      image = np.array(PIL.Image.open(io.BytesIO(row['image']['bytes'])))
      if len(image.shape) != 3:
        continue  # Filter out one bad image.
      countbenchqa_dict = {
          'image': image,
          'image_id': idx,
          'question': row['question'],
          'text': row['text'],
          'image_url': row['image_url'],
          'number': row['number'],
      }
      yield idx, countbenchqa_dict

  def _generate_examples(self, split: str):
    if split == 'huggingface':
      yield from self._generate_examples_hf()
    else:
      raise ValueError(f'Unknown split: {split}')
