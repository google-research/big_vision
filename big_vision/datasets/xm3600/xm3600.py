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
r"""Generates XM3600 in a TFDS-ready structure.

First, download the captions from https://google.github.io/crossmodal-3600/ and the images from https://cocodataset.org/#download.
The coco Karpathy split is available at http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip:
  mkdir -p /tmp/data/xm3600
  wget https://google.github.io/crossmodal-3600/web-data/captions.zip -P /tmp/data/xm3600
  unzip /tmp/data/xm3600/captions.zip -d /tmp/data/xm3600/
  wget https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz ta-P /tmp/data/xm3600
  mkdir /tmp/data/xm3600/images
  tar -xzf /tmp/data/xm3600/images.tgz -C /tmp/data/xm3600/images

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=xm3600

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load(
      'xm3600', split='en',
      data_dir='/tmp/tfds')
"""

import json
import os.path

import tensorflow_datasets as tfds

_DESCRIPTION = """
COCO image + captions, translated from English to 35 languages (English incl.).
"""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{thapliyal-etal-2022-crossmodal,
    title = "Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset",
    author = "Thapliyal, Ashish V.  and
      Pont Tuset, Jordi  and
      Chen, Xi  and
      Soricut, Radu",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.45",
    doi = "10.18653/v1/2022.emnlp-main.45",
    pages = "715--729",
}
"""
# pylint: enable=line-too-long


_CAPTIONS_PATH = '/tmp/data/xm3600'
_IMAGES_PATH = '/tmp/data/xm3600/images'

XM3600_LANGUAGES = [
    'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr',
    'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'mi', 'nl', 'no', 'pl',
    'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh'
]


class Xm3600(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for XM3600 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'First release.'}

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image/id': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'caption': tfds.features.Text(),
            'language': tfds.features.Text(),
        }),
        supervised_keys=None,
        homepage='https://google.github.io/crossmodal-3600/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {lang: self._generate_examples(lang) for lang in XM3600_LANGUAGES}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples from dataset."""
    language = split

    annot_fname = os.path.join(_CAPTIONS_PATH, 'captions.jsonl')
    data = {}
    with open(annot_fname, 'r') as f:
      for line in f:
        j = json.loads(line)
        image_id = f'{j["image/key"]}_{language}'
        captions = j[language]['caption']
        data[image_id] = captions

    for image_id, captions in data.items():
      yield image_id, {
          'image/id': image_id,
          'image': os.path.join(_IMAGES_PATH, f'{image_id.split("_")[0]}.jpg'),
          'captions': captions,
          'language': language,
      }
