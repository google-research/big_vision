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
r"""Generates COCO-35L in a TFDS-ready structure.

First, download the captions from https://google.github.io/crossmodal-3600/ and the images from https://cocodataset.org/#download.
The coco Karpathy split is available at http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip:
  mkdir -p /tmp/data/coco35l/images
  wget https://storage.googleapis.com/crossmodal-3600/coco_mt_train.jsonl.bz2 -P /tmp/data/coco35l
  wget https://storage.googleapis.com/crossmodal-3600/coco_mt_dev.jsonl.bz2 -P /tmp/data/coco35l
  bzip2 -dk  /tmp/data/coco35l/coco_mt_train.jsonl.bz2 /tmp/data/coco35l/coco_mt_dev.jsonl.bz2
  wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -P /tmp/data/coco35l
  unzip /tmp/data/coco35l/caption_datasets.zip -d /tmp/data/coco35l/
  wget http://images.cocodataset.org/zips/train2014.zip -P /tmp/data/coco35l/images
  wget http://images.cocodataset.org/zips/val2014.zip -P /tmp/data/coco35l/images
  unzip /tmp/data/coco35l/images/train2014.zip -d /tmp/data/coco35l/images/
  unzip /tmp/data/coco35l/images/val2014.zip -d /tmp/data/coco35l/images/

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=coco35l

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load(
      'coco35l', split='dev_en',
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


_CAPTIONS_PATH = '/tmp/data/coco35l'
_IMAGES_PATH = '/tmp/data/mscoco/images'
_COCOCAPS_PATH = '/tmp/data/mscoco/dataset_coco.json'

LANGUAGES = [
    'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr',
    'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'mi', 'nl', 'no', 'pl',
    'pt', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh',
]


class Coco35l(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for COCO-35L dataset."""

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
            'captions': tfds.features.Sequence(tfds.features.Text()),
            'language': tfds.features.Text(),
        }),
        supervised_keys=None,
        homepage='https://google.github.io/crossmodal-3600/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    splits = []
    for lang in LANGUAGES:
      splits.extend([f'train_{lang}', f'dev_{lang}'])
    return {split: self._generate_examples(split) for split in splits}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples from dataset."""
    split, language = split.split('_')

    id_to_path = dict()
    with open(_COCOCAPS_PATH, 'r') as f:
      data = json.load(f)['images']
    for d in data:
      id_to_path[d['cocoid']] = os.path.join(
          _IMAGES_PATH, d['filepath'], d['filename']
      )

    annot_fname = os.path.join(_CAPTIONS_PATH, f'coco_mt_{split}.jsonl')
    data = {}
    with open(annot_fname, 'r') as f:
      for line in f:
        j = json.loads(line)
        image_id = f'{j["image_id"].split("_")[0]}_{language}'
        if image_id not in data:
          data[image_id] = []
        if language == 'en':
          # COCO-35L was constructed from English into 35 other languages.
          # To add English in our TFDS, we just select a language (eg. "de") to
          # have each unique example, and add the corresponding source caption.
          if j['trg_lang'] == 'de':
            data[image_id].append(j['caption_tokenized'])
        else:
          if j['trg_lang'] == language:
            data[image_id].append(j['translation_tokenized'])

    for image_id, captions in data.items():
      yield image_id, {
          'image/id': image_id,
          'image': id_to_path[int(image_id.split('_')[0])],
          'captions': captions,
          'language': language,
      }
