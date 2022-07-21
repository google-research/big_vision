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

"""Evaluation for NYU depth.

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

import big_vision.evaluators.proj.uvim.common as common
import big_vision.pp.builder as pp_builder
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

EVAL_CROP_H = 426
EVAL_CROP_W = 560


class Evaluator:
  """Evaluator for NYU depth."""

  def __init__(self,
               predict_fn,
               pp_fn,
               batch_size,
               dataset,
               split,
               min_depth=1e-3,
               max_depth=10,
               dataset_dir=None,
               predict_kwargs=None):
    self.min_depth = min_depth
    self.max_depth = max_depth

    def predict(params, batch):
      pred = predict_fn(params, batch, **(predict_kwargs or {}))

      return jax.lax.all_gather({
          "mask": batch["mask"],
          "gt": jnp.squeeze(batch["ground_truth"], axis=-1),
          "y": pred["depth"],
      }, axis_name="data", axis=0)

    self.predict_fn = jax.pmap(predict, axis_name="data")

    # Prepare data for each process and pad with zeros so all processes have the
    # same number of batches.
    def preprocess(example):
      return {
          "mask": tf.constant(1),
          **pp_builder.get_preprocess_fn(pp_fn)(example),
      }

    self.process_batch_size = batch_size // jax.process_count()

    self.data = common.get_jax_process_dataset(
        dataset=dataset,
        dataset_dir=dataset_dir,
        split=split,
        global_batch_size=batch_size,
        pp_fn=preprocess)

  def run(self, params):
    """Run eval."""
    # Assumes that the ground truth is processed by the eval crop.
    eval_mask = np.ones((EVAL_CROP_H, EVAL_CROP_W), dtype=np.bool_)
    rmses = []
    abs_res = []
    abs_logs = []
    d1s = []
    d2s = []
    d3s = []
    for batch in self.data.as_numpy_iterator():
      # Outputs is a dict with values shaped (gather/same, devices, batch, ...)
      out = self.predict_fn(params, batch)

      if jax.process_index():  # Host0 gets all preds and does eval.
        continue

      # First, we remove the "gather" dim and transfer the result to host,
      # leading to numpy arrays of (devices, device_batch, ...)
      out = jax.tree_map(lambda x: jax.device_get(x[0]), out)
      # Then the bool-indexing with mask resulting in flat (global_batch, ...)
      out = jax.tree_map(lambda x: x[out["mask"] == 1], out)  # pylint:disable=cell-var-from-loop

      for gt, pred in zip(out["gt"], out["y"]):
        pred = _resize_nearest(pred, (EVAL_CROP_H, EVAL_CROP_W))
        valid_mask = np.logical_and(gt > self.min_depth, gt < self.max_depth)
        valid_mask = np.logical_and(valid_mask, eval_mask)

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


@functools.partial(jax.jit, static_argnums=(1,), backend="cpu")
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
