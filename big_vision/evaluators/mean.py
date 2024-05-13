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

"""Evaluator for computing mean of per-example metrics.

This evaluator can be used in two ways:
  1. Create a new evaluator with reduced boilerplate by inheriting from it.
  2. For quick prototyping, use this with predict_fns which return the metrics.
"""
from functools import partial
from typing import Mapping

from big_vision.evaluators import common

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
  for y in jax.tree.leaves(metrics):
    if y.shape != mask.shape:
      raise ValueError(
          f'Expected per-example metrics of shape {mask.shape} found '
          f'{jax.tree.map(lambda x: x.shape, metrics)}.')
  metrics = {**metrics, '_mask': mask}
  return jax.tree.map(lambda x: jnp.sum(jnp.where(mask, x, 0)), metrics)


class Evaluator:
  """Report the mean of per-example metrics computed by predict_fn.

  `predict_fn(params, batch)` must return a dict from metric name to
  per-example metrics of shape [batch_size].
  """

  def __init__(self, predict_fn, **kw):
    self.get_data_iter, self.steps = common.eval_input_pipeline(**kw)
    self.predict_fn = partial(_run_predict_fn, predict_fn)

  def run(self, train_state):
    """Computes all metrics."""
    metrics = []

    # Compute batch metrics without blocking.
    for _, batch in zip(range(self.steps), self.get_data_iter()):
      batch_metrics = self.predict_fn(train_state, batch)
      metrics.append(batch_metrics)

    # Transfer metrics (blocking).
    metrics = jax.device_get(metrics)

    # Accumulate metrics across batches.
    metrics_sum = jax.tree.map(lambda *x: np.sum(x), *metrics)
    mask_sum = metrics_sum.pop('_mask')
    for key, value_sum in metrics_sum.items():
      yield (key, value_sum / mask_sum)
