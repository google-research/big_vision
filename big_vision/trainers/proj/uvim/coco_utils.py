# Copyright 2022 Big Vision Authors.
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

"""Utilities to inspect coco data and predictions in notebooks."""
# pylint: disable=consider-using-from-import
import functools
import json

import numpy as np
from panopticapi import utils as pycoco_utils
from skimage import segmentation

import tensorflow.io.gfile as gfile


import os
ROOT = os.environ.get('COCO_DATA_DIR', '.')


PANOPTIC_COCO_CATS_FILE = f'{ROOT}/panoptic_coco_categories.json'


@functools.lru_cache(maxsize=None)
def _coco_panoptic_categories():
  with gfile.GFile(PANOPTIC_COCO_CATS_FILE, 'r') as f:
    categories_list = json.load(f)
  return tuple(categories_list)


def rgb_panoptic_from_twochannels(twochannels, boundaries: bool = False):
  """Makes a RGB panoptic output and segments_info from a twochannels view."""
  semantics = twochannels[..., 0]
  instances = twochannels[..., 1]
  max_instances = np.max(instances) + 1
  merged = semantics * max_instances + instances
  merged = np.where(semantics < 0, semantics, merged)

  categories_list = _coco_panoptic_categories()
  categories = {category['id']: category for category in categories_list}
  id_generator = pycoco_utils.IdGenerator(categories)
  segments_info = {}
  rgb = np.zeros((*instances.shape[:2], 3), dtype=np.uint8)

  for merged_id in np.unique(merged):
    if merged_id // max_instances > 0:
      category = categories_list[int(merged_id // max_instances) - 1]
      segment_id, color = id_generator.get_id_and_color(category['id'])
    else:
      category = {'id': -1, 'name': 'void', 'isthing': False}
      segment_id, color = -1, np.array([0, 0, 0])
    segments_info[segment_id] = {
        'id': segment_id,
        'color': color,
        'category_id': category['id'],
        'name': category['name'],
        'isthing': category['isthing'],
    }
    rgb[merged == merged_id] = color

  if boundaries:
    boundaries = segmentation.find_boundaries(
        pycoco_utils.rgb2id(rgb), mode='thick')
    rgb[boundaries] = 0
  return rgb, segments_info
