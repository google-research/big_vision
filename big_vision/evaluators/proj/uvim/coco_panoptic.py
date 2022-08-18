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

"""COCO17 panoptic evaluation."""
import functools
from functools import partial
import json
import os
import tempfile
import time
import zipfile

from absl import logging
from big_vision.evaluators.proj.uvim import common
import big_vision.pp.builder as pp_builder
import jax
import numpy as np
import panopticapi_converters.twochannels2panoptic_coco_format as converter
from panopticapi.evaluation import pq_compute
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.io import gfile


ROOT = os.environ.get('COCO_DATA_DIR', '.')

PANOPTIC_COCO_CATS_FILE = f'{ROOT}/panoptic_coco_categories.json'
PANOPTIC_2017 = {
    'train': f'{ROOT}/panoptic_train2017.json',
    'validation': f'{ROOT}/panoptic_val2017.json',
}

PANOPTIC_GT_ZIP = {
    'train': f'{ROOT}/panoptic_train2017.zip',
    'validation': f'{ROOT}/panoptic_val2017.zip',
}


class Evaluator:
  """Panoptic segmentation evaluator: calls official COCO API.

  `predict_fn` accepts arbitrary dictionaries of parameters and data, where
  the data dictionary is produced by the `pp` op. It is expected to output a
  2-channel mask, where the first channel encodes semantics, and the second
  channel encodes instance ids.
  """

  def __init__(self,
               predict_fn,
               pp_fn,
               batch_size,
               dataset='coco/2017_panoptic',
               dataset_dir=None,
               split='validation',
               predict_kwargs=None):
    # Prepare to run predict on all processes and gather predictions on all
    # devices. Note: if needed consider only gather across processes.
    def predict(params, batch):
      res = {
          'image/id': batch['image/id'],
          'mask': batch['mask'],
          'y': predict_fn(params, batch['input'], **(predict_kwargs or {})),
      }
      return jax.lax.all_gather(res, axis_name='data', axis=0)

    self.predict_fn = jax.pmap(predict, axis_name='data')

    # Prepare data for each process and pad with zeros so all processes have the
    # same number of batches.
    def preprocess(example):
      return {
          'image/id': example['image/id'],
          'mask': tf.constant(1),
          'input': pp_builder.get_preprocess_fn(pp_fn)(example),
      }

    self.data = common.get_jax_process_dataset(
        dataset, split, dataset_dir=dataset_dir,
        global_batch_size=batch_size,
        pp_fn=preprocess)

    # Only process 0 runs conversion to png and calls into coco api.
    if jax.process_index() == 0:
      self.result_dir = tempfile.TemporaryDirectory()
      (self.gt_folder, self.gt_json, self.categories_json,
       self.remap, self.size_map) = _prepare_ground_truth(
           dataset, split, dataset_dir)

  def _compute_png_predictions(self, params):
    """Computes predictions and converts then to png to optimize memory use."""
    count = 0
    logging.info('Panoptic eval: running inference.')
    for batch in self.data.as_numpy_iterator():
      out = self.predict_fn(params, batch)

      if jax.process_index():
        continue

      out = jax.device_get(jax.tree_map(lambda x: x[0], out))
      mask = out['mask']
      pan_recs = out['y'][mask != 0]
      ids = out['image/id'][mask != 0]

      for pan_rec, image_id in zip(pan_recs, ids):
        sem = pan_rec[..., 0]
        ins = pan_rec[..., 1]

        sem_remapped = np.array(sem)
        for v in np.unique(sem):
          sem_remapped[sem == v] = self.remap[v]
        sem = sem_remapped

        pan_mask = np.stack([sem, ins, np.zeros_like(sem)], axis=-1)
        pan_mask = _resize_nearest(pan_mask, self.size_map[image_id])
        pan_mask_png = tf.io.encode_png(pan_mask.astype('uint8')).numpy()

        fname = f'{self.result_dir.name}/{image_id:012d}.png'
        with open(fname, 'wb') as f:
          f.write(pan_mask_png)
        count += 1

      logging.log_every_n_seconds(
          logging.INFO, 'Panoptic eval: processed %i examples so far.', 30,
          count)

    if jax.process_index():
      return None

    logging.info('Panoptic eval: inference done. Processed %d examples.', count)
    return self.result_dir

  def run(self, params):
    """Run eval."""
    # Note result_dir is constant, but files inside are mutated.
    result_dir = self._compute_png_predictions(params)

    if not result_dir:
      return

    with tempfile.TemporaryDirectory() as pred_folder, \
         tempfile.NamedTemporaryFile(mode='w') as pred_json:

      logging.info('Panoptic eval: running conversion.')
      converter.converter(
          source_folder=result_dir.name,
          images_json_file=self.gt_json,
          categories_json_file=self.categories_json,
          segmentations_folder=pred_folder,
          predictions_json_file=pred_json.name)
      logging.info('Panoptic eval: conversion done.')

      logging.info('Panoptic eval: running metrics computation.')
      res = pq_compute(gt_json_file=self.gt_json,
                       gt_folder=self.gt_folder,
                       pred_json_file=pred_json.name,
                       pred_folder=pred_folder)
      logging.info('Panoptic eval: metrics computation done.')

    for k in ['All', 'Stuff', 'Things']:
      for m in ['pq', 'rq', 'sq']:
        yield f'{k}_{m}', res[k][m]


def _prepare_ground_truth(dataset, split, data_dir):
  """Prepare ground truth from tf.data.Dataset."""
  if dataset == 'coco/2017_panoptic' and data_dir is None:
    return _prepare_ground_truth_from_zipfiles(split)
  else:
    return _prepare_ground_truth_from_dataset(dataset, split, data_dir)


@functools.lru_cache(maxsize=None)
def _prepare_ground_truth_from_dataset(dataset, split, data_dir):
  """Prepare ground truth from a tf.data.Dataset."""
  dataset = tfds.builder(dataset, data_dir=data_dir).as_dataset(split=split)

  categories_json = _make_local_copy(PANOPTIC_COCO_CATS_FILE)
  with gfile.GFile(categories_json, 'rb') as f:
    categories = json.loads(f.read())

  # Build map from tfds class ids to COCO class ids.
  remap = {0: 0}
  with gfile.GFile(categories_json, 'r') as f:
    remap = {**remap, **{(i + 1): x['id'] for i, x in enumerate(categories)}}

  gt_folder = tempfile.mkdtemp()
  gfile.makedirs(gt_folder)
  size_map = {}
  annotations = []
  images = []
  for example in dataset:
    image_id = int(example['image/id'])
    panoptic_image = example['panoptic_image']
    ann_ids = example['panoptic_objects']['id']
    ann_labels = example['panoptic_objects']['label']
    ann_iscrowd = example['panoptic_objects']['is_crowd']
    ann_area = example['panoptic_objects']['area']

    fname = f'{image_id:012d}.png'
    with gfile.GFile(os.path.join(gt_folder, fname), 'wb') as f:
      f.write(tf.io.encode_png(panoptic_image).numpy())

    size_map[image_id] = (panoptic_image.shape[0], panoptic_image.shape[1])

    segments_info = []
    for i in range(len(ann_ids)):
      segments_info.append({
          'id': int(ann_ids[i]),
          'category_id': remap[int(ann_labels[i] + 1)],
          'iscrowd': int(ann_iscrowd[i]),
          'area': int(ann_area[i]),
      })

    annotations.append({
        'file_name': str(fname),
        'image_id': int(image_id),
        'segments_info': segments_info
    })
    images.append({
        'id': image_id,
        'file_name': f'{image_id:012d}.jpg',
    })

  # Write annotations.json needed for pq_compute.
  gt_json = os.path.join(gt_folder, 'annotations.json')
  with gfile.GFile(gt_json, 'wb') as f:
    f.write(json.dumps({
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }))

  return gt_folder, gt_json, categories_json, remap, size_map


def _prepare_ground_truth_from_zipfiles(split):
  """Prepare ground truth from coco zip files."""
  split_prefix = split.split('[')[0]
  if split_prefix not in ('train', 'validation'):
    raise ValueError(f'Split {split} not supported')

  # The following 4 calls are cached. This allows to save significant time
  # in use cases like sweeping predict_fn hparams on the same run.
  gt_json = _make_local_copy(PANOPTIC_2017[split_prefix])
  gt_folder = _make_local_unzip_copy(PANOPTIC_GT_ZIP[split_prefix])
  categories_json = _make_local_copy(PANOPTIC_COCO_CATS_FILE)
  image_ids = _list_image_ids('coco/2017_panoptic', split)

  gt_folder = os.path.join(
      gt_folder, 'panoptic_val2017'
      if split_prefix == 'validation' else 'panoptic_train2017')

  # Build map from tfds class ids to COCO class ids.
  remap = {0: 0}
  with gfile.GFile(categories_json, 'r') as f:
    remap = {**remap, **{(i + 1): x['id'] for i, x in enumerate(json.load(f))}}

  # Filters gt_json to contain only annotations for images in dataset.
  with gfile.GFile(gt_json) as f:
    data = json.load(f)
  logging.info(
      'Panoptic eval: pre-filter %d annotations.',
      len(data['annotations'])
  )
  data['images'] = [x for x in data['images'] if x['id'] in image_ids]
  data['annotations'] = [
      x for x in data['annotations'] if x['image_id'] in image_ids
  ]
  logging.info(
      'Panoptic eval: post-filter %d annotations.',
      len(data['annotations'])
  )
  filtered_gt_json = tempfile.NamedTemporaryFile(delete=False).name
  with open(filtered_gt_json, 'w') as f:
    json.dump(data, f)

  # Precompute images sizes.
  size_map = {x['id']: (x['height'], x['width']) for x in data['images']}

  return gt_folder, filtered_gt_json, categories_json, remap, size_map


@functools.lru_cache(maxsize=None)
def _list_image_ids(dataset, split):
  d = tfds.load(dataset, split=split).map(lambda x: x['image/id'])
  return frozenset(d.as_numpy_iterator())


@functools.lru_cache(maxsize=None)
def _make_local_copy(fname) -> str:
  start = time.monotonic()
  local_file = tempfile.NamedTemporaryFile(delete=False)
  gfile.copy(fname, local_file.name, overwrite=True)
  logging.info('Copy %s in %d seconds.', fname, time.monotonic() - start)
  return local_file.name


@functools.lru_cache(maxsize=None)
def _make_local_unzip_copy(fname) -> str:
  start = time.monotonic()
  folder = tempfile.mkdtemp()
  with tempfile.NamedTemporaryFile() as tmp_zip_file:
    gfile.copy(fname, tmp_zip_file.name, overwrite=True)
    with zipfile.ZipFile(tmp_zip_file.name, 'r') as f:
      f.extractall(folder)
  logging.info('Copy %s in %d seconds.', fname, time.monotonic() - start)
  return folder


@partial(jax.jit, static_argnums=(1,), backend='cpu')
def _resize_nearest(image, shape):
  return jax.image.resize(image, shape + image.shape[-1:], 'nearest')
