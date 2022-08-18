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

"""Evaluator for computing mean of per-example metrics."""
import functools
from typing import Mapping

from big_vision import input_pipeline
from big_vision.datasets import core as ds_core
from big_vision.pp import builder as pp_builder

import jax
import jax.numpy as jnp
import numpy as np


# Note: global to avoid jax re-compiling across different evaluator instances.
@functools.partial(jax.pmap, static_broadcasted_argnums=0, axis_name='batch')
def _run_predict_fn(predict_fn, params, batch):
  """Sum per-example metrics weighted by `_mask`."""
  mask = batch['_mask']
  metrics = predict_fn(params, batch)
  # Sanity check output format of predict_fn.
  assert isinstance(metrics, Mapping), 'predict_fn must return a dict'
  for y in jax.tree_leaves(metrics):
    if y.shape != mask.shape:
      raise ValueError(
          f'Expected per-example metrics of shape {mask.shape} found '
          f'{jax.tree_map(lambda x: x.shape, metrics)}.')
  metrics = {**metrics, '_mask': mask}
  metrics = jax.tree_map(lambda x: jnp.inner(x, mask), metrics)
  return jax.lax.psum(metrics, axis_name='batch')


class Evaluator:
  """Report the mean of per-example metrics computed by predict_fn.

  `predict_fn(params, batch)` must return a dict from metric name to
  per-example metrics of shape [batch_size].
  """

  def __init__(self, predict_fn, data, pp_fn, batch_size,
               cache_final=True, cache_raw=False, prefetch=1):
    data = ds_core.get(**data)
    self.dataset, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), batch_size=batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        preprocess_fn=pp_builder.get_preprocess_fn(pp_fn),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_input_pipeline(self.dataset, prefetch)
    self.predict_fn = predict_fn

  def run(self, params):
    """Computes all metrics."""
    metrics = []

    # Compute batch metrics without blocking.
    for _, batch in zip(range(self.steps), self.data_iter):
      batch_metrics = _run_predict_fn(self.predict_fn, params, batch)
      metrics.append(batch_metrics)

    # Transfer metrics from device 0 to host (blocking).
    metrics = jax.device_get(jax.tree_map(lambda x: x[0], metrics))

    metrics_sum = jax.tree_map(lambda *x: np.sum(x), *metrics)
    mask_sum = metrics_sum.pop('_mask')
    for key, value_sum in metrics_sum.items():
      yield (key, value_sum / mask_sum)
