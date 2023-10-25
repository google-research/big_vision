# Copyright 2023 Big Vision Authors.
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

"""Evaluator for computing mean of per-example metrics.

This evaluator can be used in two ways:
  1. Create a new evaluator with reduced boilerplate by inheriting from it.
  2. For quick prototyping, use this with predict_fns which return the metrics.
"""
from functools import partial
from typing import Mapping

from big_vision import input_pipeline
from big_vision.datasets import core as ds_core
from big_vision.pp import builder as pp_builder

import jax
import jax.numpy as jnp
import numpy as np


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# Note: global to avoid jax re-compiling across different evaluator instances.
@partial(jax.jit, static_argnums=0)
def _run_predict_fn(predict_fn, train_state, batch):
  """Sum per-example metrics weighted by `_mask`."""
  mask = batch['_mask']
  metrics = predict_fn(train_state, batch)
  # Sanity check output format of predict_fn.
  assert isinstance(metrics, Mapping), 'predict_fn must return a dict'
  for y in jax.tree_leaves(metrics):
    if y.shape != mask.shape:
      raise ValueError(
          f'Expected per-example metrics of shape {mask.shape} found '
          f'{jax.tree_map(lambda x: x.shape, metrics)}.')
  metrics = {**metrics, '_mask': mask}
  return jax.tree_map(lambda x: jnp.sum(jnp.where(mask, x, 0)), metrics)


class Evaluator:
  """Report the mean of per-example metrics computed by predict_fn.

  `predict_fn(params, batch)` must return a dict from metric name to
  per-example metrics of shape [batch_size].
  """

  def __init__(self, predict_fn, data, pp_fn, batch_size,
               cache_final=True, cache_raw=False, prefetch=1, *, devices):
    data = ds_core.get(**data)
    self.dataset, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), batch_size=batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        preprocess_fn=pp_builder.get_preprocess_fn(pp_fn),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(
        self.dataset, devices, prefetch)
    self.predict_fn = partial(_run_predict_fn, predict_fn)

  def run(self, train_state):
    """Computes all metrics."""
    metrics = []

    # Compute batch metrics without blocking.
    for _, batch in zip(range(self.steps), self.data_iter):
      batch_metrics = self.predict_fn(train_state, batch)
      metrics.append(batch_metrics)

    # Transfer metrics (blocking).
    metrics = jax.device_get(metrics)

    # Accumulate metrics across batches.
    metrics_sum = jax.tree_map(lambda *x: np.sum(x), *metrics)
    mask_sum = metrics_sum.pop('_mask')
    for key, value_sum in metrics_sum.items():
      yield (key, value_sum / mask_sum)
