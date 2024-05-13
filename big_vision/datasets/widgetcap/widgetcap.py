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
r"""Import widgetcap into TFDS format.

  Widget Captioning all requires images from the RICO dataset:
  mkdir -p /tmp/data/rico_images ; cd /tmp/data/rico_images
  wget
  https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
  tar xvfz unique_uis.tar.gz
  rm unique_uis.tar.gz

  Widget Captioning:
  mkdir - /tmp/data/widget_captioning ; cd /tmp/data/widget_captioning
  git clone https://github.com/google-research-datasets/widget-caption.git
  cp widget-caption/widget_captions.csv ./
  cp widget-caption/split/*.txt ./
  rm -rf widget-caption

Then, run conversion locally (make sure to install tensorflow-datasets for the
`tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=widgetcap

Example to load:

  import tensorflow_datasets as tfds
  dataset_augmented = tfds.load('widgetcap', split='train',
  data_dir='/tmp/tfds')
"""
import csv
import json
import os

import numpy as np
from PIL import Image
import tensorflow_datasets as tfds

_DATASET_DIR = '/tmp/data/widget_captioning'
# Dataset property indicating the y-dim of the canvas
_RICO_CANVAS_Y = 2560
_IMAGE_DIR = '/tmp/data/rico_images/combined'

_CITATION = (
    '@inproceedings{Li2020WidgetCG,title={Widget Captioning: Generating Natural'
    ' Language Description for MobileUser Interface Elements},author={Y. Li and'
    ' Gang Li and Luheng He and Jingjie Zheng and Hong Li and Zhiwei'
    ' Guan},booktitle={Conference on Empirical Methods in Natural Language'
    ' Processing},year={2020},}'
)


class Widgetcap(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for widgetcap dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Format as needed for PaliGemma'}

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description='The widgetcap dataset.',
        features=tfds.features.FeaturesDict({
            'image/id': tfds.features.Text(),
            'image/filename': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'texts': tfds.features.Sequence(tfds.features.Text()),
            'bbox': tfds.features.BBoxFeature(),
            'screen_id': tfds.features.Text(),
            'node_id': tfds.features.Text(),
            'height': np.int32,
            'width': np.int32,
        }),
        homepage='https://github.com/google-research-datasets/widget-caption',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
        'train': self._generate_examples('train'),
        'dev': self._generate_examples('dev'),
        'test': self._generate_examples('test'),
    }

  def _generate_examples(self, split):
    """Yields (key, example) tuples from the dataset."""
    split_screen_ids = set()
    with open(os.path.join(_DATASET_DIR, split + '.txt')) as f:
      for line in f:
        split_screen_ids.add(line.strip())

    with open(os.path.join(_DATASET_DIR, 'widget_captions.csv')) as f:
      for row in csv.DictReader(f):
        if row['screenId'] in split_screen_ids:
          id_, example = self._get_example(
              row['screenId'], row['nodeId'], row['captions']
          )
          yield id_, example

  def _get_node_box(self, screen_id, node_id, height):
    index_list = [int(i) for i in node_id.split('.')[1:]]
    with open(os.path.join(_IMAGE_DIR, screen_id + '.json')) as f:
      view = json.load(f)
    curr_node = view['activity']['root']
    for index in index_list:
      curr_node = curr_node['children'][index]
    normalized_bounds = map(
        lambda x: x * height / _RICO_CANVAS_Y, curr_node['bounds']
    )
    return normalized_bounds

  def _get_example(self, screen_id, node_id, captions):
    image = Image.open(os.path.join(_IMAGE_DIR, screen_id + '.jpg'))
    width, height = image.size
    # get bounding box coordinates
    xmin, ymin, xmax, ymax = self._get_node_box(screen_id, node_id, height)

    image_id = f'{screen_id}_{node_id}'
    example = {
        'image/id': image_id,
        'image/filename': screen_id + '.jpg',
        'image': os.path.join(_IMAGE_DIR, screen_id + '.jpg'),
        'texts': captions.split('|'),
        'bbox': tfds.features.BBox(
            ymin=ymin / height,
            xmin=xmin / width,
            ymax=ymax / height,
            xmax=xmax / width,
        ),
        'screen_id': screen_id,
        'node_id': node_id,
        'height': height,
        'width': width,
    }
    return image_id, example
