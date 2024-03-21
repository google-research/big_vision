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

"""Evaluation for NYU depth.

jax.jit-compatible fork of the evaluator from evaluators/proj/uvim.

At evaluation time the ground truth is cropped and clipped. Values outside of
the test crop or clipping range are not included in eval calculations.

In this evaluator, it is assume that the groud truth is already cropped, so the
entire image is evaluated. However, the evaluator does perform the clipping.

Reference implementations:
  https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blo(internal link)a0f341244260ff61541191a613dd74bc/depth/datasets/nyu.py
  https://github.com/vinvino02/GLPDepth/blob/7f3c78df4ecd6e7c79fd0c4b73c95d61f4aa2121/code/utils/metrics.py
  https://github.com/shariqfarooq123/AdaBins/blob/2fb686a66a304f0a719bc53d77412460af97fd61/evaluate.py
"""

import functools
import itertools

from big_vision import input_pipeline
from big_vision import utils
from big_vision.datasets import core as ds_core
import big_vision.pp.builder as pp_builder
import jax
import jax.numpy as jnp
import numpy as np

# Temporary global flag to facilitate backwards compatability.
API = "jit"


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
    pred = predict_fn(train_state, batch)
    return {"mask": batch["_mask"],
            "gt": jnp.squeeze(batch["ground_truth"], axis=-1),
            "y": pred["depth"]}
  return _run_predict_fn


class Evaluator:
  """Evaluator for NYU depth."""

  def __init__(self,
               predict_fn,
               pp_fn,
               batch_size,
               data,
               cache_final=True,
               cache_raw=False,
               prefetch=1,
               min_depth=1e-3,
               max_depth=10,
               *,
               devices):
    """Evaluator for NYU depth.

    Args:
      predict_fn: jit-compilable function, accepts arbitrary dictionaries of 
        parameters and data, where the data dictionary is produced by the
        `pp_fn` op. It is expected to output a dict with `depth` containing an
        2D array with the predicted depth. The prediction is resized to the
        ground_truth size with nearest neighbour.
      pp_fn: Preprocessing function, sepcified as string. `pp_fn` must also
        output a 'ground_truth' as a 2D array of ground truth. Fruther, it has
        to apply a crop, if one wants to compute metrics with the eval crop
        typically used for NYU Depth metrics.
      batch_size: Batch size.
      data: Dict specifying name and split of the data set. Defaults to the
        standard COCO (2017).
      cache_final: Whether to cache the data after preprocessing - see
        input_pipeline for details.
      cache_raw: Whether to cache the raw data - see input_pipline for details.
      prefetch: Number of batches to prefetch
      min_depth: Minimum depth value.
      max_depth: Maximum depth value.
      devices: List of jax devices.
    """
    self.min_depth = min_depth
    self.max_depth = max_depth
    self.predict_fn = _get_predict_fn(
        predict_fn, jax.sharding.Mesh(devices, ("devices",)))

    data = ds_core.get(**data)
    self.dataset, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), batch_size=batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        preprocess_fn=pp_builder.get_preprocess_fn(pp_fn),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(
        self.dataset, devices, prefetch)

  def run(self, train_state):
    """Run NYU depth eval.

    Args:
      train_state: pytree containing the model parameters.

    Yields:
      Tuples consisting of metric name and value.
    """
    rmses = []
    abs_res = []
    abs_logs = []
    d1s = []
    d2s = []
    d3s = []
    for batch in itertools.islice(self.data_iter, self.steps):
      # Outputs is a dict with values shaped (gather/same, devices, batch, ...)
      out = self.predict_fn(train_state, batch)

      if jax.process_index():  # Host0 gets all preds and does eval.
        continue

      out = jax.device_get(out)
      # Then the bool-indexing with mask resulting in flat (global_batch, ...)
      out = jax.tree_map(lambda x: x[out["mask"]], out)  # pylint:disable=cell-var-from-loop

      for gt, pred in zip(out["gt"], out["y"]):
        # put_cpu and conversion to numpy arrays below to avoid unwanted
        # host-to-device transfers
        pred, gt = utils.put_cpu((pred, gt))
        pred = _resize_nearest(pred, (gt.shape[0], gt.shape[1]))
        pred, gt = np.array(pred), np.array(gt)
        valid_mask = np.logical_and(gt > self.min_depth, gt < self.max_depth)

        rmses.append(_compute_rmse(gt[valid_mask], pred[valid_mask]))
        abs_res.append(_compute_abs_re(gt[valid_mask], pred[valid_mask]))
        abs_logs.append(_compute_abs_log(gt[valid_mask], pred[valid_mask]))
        d1s.append(_compute_delta(gt[valid_mask], pred[valid_mask], order=1))
        d2s.append(_compute_delta(gt[valid_mask], pred[valid_mask], order=2))
        d3s.append(_compute_delta(gt[valid_mask], pred[valid_mask], order=3))

    if jax.process_index():  # Host0 gets all preds and does eval.
      return

    yield "RMSE", np.mean(rmses)
    yield "abs_RE", np.mean(abs_res)
    yield "log10", np.mean(abs_logs)
    yield "delta1", np.mean(d1s)
    yield "delta2", np.mean(d2s)
    yield "delta3", np.mean(d3s)


@utils.jit_cpu(static_argnums=(1,))
def _resize_nearest(image, shape):
  return jax.image.resize(image, shape, "nearest")


def _compute_rmse(gt, pred):
  diff = gt - pred
  return np.sqrt(np.mean(np.power(diff, 2)))


def _compute_abs_re(gt, pred):
  diff = np.abs(gt - pred)
  return np.mean(diff / gt)


def _compute_abs_log(gt, pred):
  diff = np.abs(np.log10(gt) - np.log10(pred))
  return np.mean(diff)


def _compute_delta(gt, pred, order):
  rel_diff = np.maximum(gt / pred, pred / gt)
  return np.sum(rel_diff < 1.25**order) / rel_diff.size
