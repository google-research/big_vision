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
r"""Implements ST-VQA dataset in TFDS.

It's small data, so simple to run locally.
First, download and unzip the dataset from https://rrc.cvc.uab.es/?ch=11
and place it in /tmp/data/stvqa.

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd third_party/py/big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=stvqa

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('stvqa', split='train', data_dir='/tmp/tfds')

Dataset splits:
  train: 23446 examples/questions (subset of original train)
  val: 2628 examples/questions (subset of original train)
  test: 4070 examples/questions (no answers)

Note: original source data has no val/holdout split, and we therefore split the
original train split (26074 examples/questions) by ourselves into train & val
splits.

Recommended training splits:
  train: train
  minitrain: train[:5%]
  eval: val
  fulltrain: train+val
"""
import json
import os

import numpy as np
import tensorflow_datasets as tfds
import val_ids

_VAL_IDS = val_ids.PSEUDO_VAL_IMAGE_PATHS

_DESCRIPTION = """ST-VQA dataset."""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{Biten_2019,
   title={Scene Text Visual Question Answering},
   url={http://dx.doi.org/10.1109/ICCV.2019.00439},
   DOI={10.1109/iccv.2019.00439},
   booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},
   publisher={IEEE},
   author={Biten, Ali Furkan and Tito, Ruben and Mafla, Andres and Gomez, Lluis and Rusinol, Marcal and Jawahar, C.V. and Valveny, Ernest and Karatzas, Dimosthenis},
   year={2019},
   month=oct }
"""
# pylint: enable=line-too-long

# When running locally (recommended), copy files as above an use these:
_STVQA_PATH = '/tmp/data/stvqa/'


class Stvqa(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ST-VQA dataset."""

  VERSION = tfds.core.Version('1.2.0')
  RELEASE_NOTES = {
      '1.0.0': 'First release.',
      '1.1.0': 'Switch to COCO high-res images and lower-case answers.',
      '1.2.0': 'Rename pseudo splits and remove lower-case answers.',
      }

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'question_id': tfds.features.Scalar(np.int32),
            'filename': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'question': tfds.features.Text(),
            'answers': tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,
        homepage='https://rrc.cvc.uab.es/?ch=11',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split)
            for split in ('train', 'val', 'test')}

  def _generate_examples(self, split):
    """Yields (key, example) tuples."""
    src_split = 'test' if split == 'test' else 'train'
    annot_fname = os.path.join(_STVQA_PATH, f'{src_split}_task_3.json')
    images_path = f'{src_split}{"_task3" if src_split == "test" else ""}_images'

    with open(annot_fname, 'r') as f:
      data = json.loads(f.read())

    for x in data['data']:
      if split == 'val' and x['file_path'] not in _VAL_IDS:
        continue
      elif split == 'train' and x['file_path'] in _VAL_IDS:
        continue
      image_path = os.path.join(_STVQA_PATH, images_path, x['file_path'])
      # Always use high-res COCO images from train2014 directory.
      if x['file_path'].startswith('coco-text'):
        image_path = image_path.replace(os.path.join(images_path, 'coco-text'),
                                        'train2014')
      yield x['question_id'], {
          'question_id': x['question_id'],
          'filename': x['file_path'],
          'image': image_path,
          'question': x['question'],
          'answers': x.get('answers', []),
      }
