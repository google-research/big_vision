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

"""COCO17 panoptic evaluation.

jax.jit-compatible fork of the evaluator from evaluators/proj/uvim.
"""
import functools
import itertools
import json
import os
import tempfile
import time
from typing import Any
import zipfile

from absl import flags
from absl import logging
from big_vision import input_pipeline
from big_vision import utils
from big_vision.datasets import core as ds_core
import big_vision.pp.builder as pp_builder
import jax
import jax.numpy as jnp
import numpy as np
from pycocotools.panopticapi import evaluation
import panopticapi_converters.twochannels2panoptic_coco_format as converter
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.io import gfile

# Temporary global flag to facilitate backwards compatability.
API = 'jit'

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


# Note: global to avoid jax re-compiling across different evaluator instances.
@functools.cache
def _get_predict_fn(predict_fn, mesh=None):
  """Wrapper for jit-compiled predict function."""

  # `out_shardings` annotation is needed because of the `all_gather` ops in the
  # pmap implementation.
  @functools.partial(jax.jit,
                     out_shardings=jax.sharding.NamedSharding(
                         mesh, jax.sharding.PartitionSpec()))
  def _run_predict_fn(train_state, batch):
    """Run predict_fn and gather all outputs on all devices."""
    y = predict_fn(train_state, batch)
    res = {
        'image/id': batch['image/id'],
        'mask': batch['_mask'],
        'y': jnp.stack([y['semantics'], y['instances']], axis=-1),
    }
    return res
  return _run_predict_fn


class Evaluator:
  """Panoptic segmentation evaluator: calls official COCO API."""

  def __init__(
      self,
      predict_fn,
      pp_fn,
      batch_size,
      data=None,
      cache_final=True,
      cache_raw=False,
      prefetch=1,
      save_dir=None,
      *,
      devices,
  ):
    """Panoptic segmentation evaluator: calls official COCO API.

    Args:
      predict_fn: jit-compilable function, which accepts arbitrary dictionaries
        of parameters and data, where the data dictionary is produced by the
        `pp_fn`. It is expected to output a 2-channel mask, where the first
        channel encodes semantics, and the second channel encodes instance ids.
      pp_fn: Preprocessing function, sepcified as string.
      batch_size: Batch size.
      data: Dict specifying name and split of the data set. Defaults to the
        standard COCO (2017).
      cache_final: Whether to cache the data after preprocessing - see
        input_pipeline for details.
      cache_raw: Whether to cache the raw data - see input_pipline for details.
      prefetch: Number of batches to prefetch
      save_dir: Directory to save the results in.
      devices: List of jax devices.
    """
    self.predict_fn = _get_predict_fn(
        predict_fn, jax.sharding.Mesh(devices, ('devices',)))

    data_specs = dict(name='coco/2017_panoptic',
                      data_dir=None, split='validation')
    data_specs.update(data or {})
    data = ds_core.get(**data_specs)
    self.dataset, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), batch_size=batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        preprocess_fn=pp_builder.get_preprocess_fn(pp_fn),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(
        self.dataset, devices, prefetch)

    # Only process 0 runs conversion to png and calls into coco api.
    if jax.process_index() == 0:
      self.result_dir = tempfile.TemporaryDirectory()
      (self.gt_folder, self.gt_json, self.categories_json,
       self.remap, self.size_map) = _prepare_ground_truth(
           data_specs['name'], data_specs['split'],
           data_specs.get('data_dir'))
      if save_dir:
        self.save_dir = save_dir.format(workdir=flags.FLAGS.workdir)
        gfile.makedirs(self.save_dir)
      else:
        self.save_dir = None

  def _compute_png_predictions(
      self, train_state: Any) -> Any:
    """Computes predictions and converts then to png to optimize memory use."""
    count = 0
    logging.info('Panoptic eval: running inference.')
    for batch in itertools.islice(self.data_iter, self.steps):
      out = self.predict_fn(train_state, batch)

      if jax.process_index():
        continue

      out = jax.device_get(out)
      mask = out['mask']
      pan_recs = out['y'][mask]
      ids = out['image/id'][mask]

      for pan_rec, image_id in zip(pan_recs, ids):
        sem = pan_rec[..., 0]
        ins = pan_rec[..., 1]

        sem_remapped = np.array(sem)
        for v in np.unique(sem):
          sem_remapped[sem == v] = self.remap[v]
        sem = sem_remapped

        pan_mask = np.stack([sem, ins, np.zeros_like(sem)], axis=-1)
        pan_mask = utils.put_cpu(pan_mask)
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

  def run(self, train_state):
    """Run panoptic segmentation evaluation.

    Args:
      train_state: pytree containing the model parameters.

    Yields:
      Tuples consisting of metric name and value.
    """
    # Note result_dir is constant, but files inside are mutated.
    result_dir = self._compute_png_predictions(train_state)

    if jax.process_index():
      return

    if self.save_dir:
      gfile.RecursivelyCopyDir(result_dir.name, self.save_dir, overwrite=True)

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
      res = evaluation.pq_compute(gt_json_file=self.gt_json,
                                  gt_folder=self.gt_folder,
                                  pred_json_file=pred_json.name,
                                  pred_folder=pred_folder)
      logging.info('Panoptic eval: metrics computation done.')

    for k in ['All', 'Stuff', 'Things']:
      for m in ['pq', 'rq', 'sq']:
        yield f'{k}_{m}', res[k][m]


def _prepare_ground_truth(dataset, split, data_dir):
  if dataset == 'coco/2017_panoptic' and data_dir is None:
    return _prepare_ground_truth_from_zipfiles(split)
  else:
    return _prepare_ground_truth_from_dataset(dataset, split, data_dir)


@functools.lru_cache(maxsize=None)
def _prepare_ground_truth_from_dataset(dataset, split, data_dir):
  """Prepare ground truth from a tf.data.Dataset.
  
  Args:
    dataset: TFDS-compatible dataset specification.
    split: Data set split to use.
    data_dir: Folder containing the data
  
  Returns:
    A tuple containing the folder containing the ground-truth data, the
    ground truth annotations loaded from json, the categories loaded form json,
    a map for remapping, and a map mapping image id to image size.
  
  """
  tfds_dataset = tfds.builder(
      dataset, data_dir=data_dir).as_dataset(split=split)

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
  for example in tfds_dataset:
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
  """Prepare ground truth from coco zip files.

  Args:
    split: dataset split to prepare ground truth for.

  Returns:
    A tuple containing the folder containing the ground-truth data, the ground
    truth annotations loaded from json, the categories loaded form json, a map
    for remapping, and a map mapping image id to image size.
  """
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


@utils.jit_cpu(static_argnums=(1,))
def _resize_nearest(image, shape):
  return jax.image.resize(image, shape + image.shape[-1:], 'nearest')
