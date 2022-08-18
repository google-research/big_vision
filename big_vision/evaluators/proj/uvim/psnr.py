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

"""Compute PSNR, currently used for colorization and superresolution."""

import functools

import big_vision.evaluators.proj.uvim.common as common
import big_vision.pp.builder as pp_builder
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


class Evaluator:
  """PSNR evaluator.

  `predict_fn` accepts arbitrary dictionaries of parameters and data, where
  the data dictionary is produced by the `pp_fn` op. It is expected to output a
  single-key dict containing an RGB image with intensities in [-1,1].
  """

  def __init__(self,
               predict_fn,
               pp_fn,
               batch_size,
               dataset="imagenet2012",
               split="validation",
               predict_kwargs=None):

    def predict(params, batch):

      def _f(x):
        y = predict_fn(params, x, **(predict_kwargs or {}))
        # Assume image intensities are in [-1,1].
        # Evaluator expects a dict with a single item.
        pred, = y.values()
        return _psnr(pred, x["labels"], 2.)
      return jax.lax.all_gather({
          "mask": batch["mask"],
          "psnr": _f(batch["input"]),
      }, axis_name="data", axis=0)

    self.predict_fn = jax.pmap(predict, axis_name="data")

    # Prepare data for each process and pad with zeros so all processes have the
    # same number of batches.
    def preprocess(example):
      return {
          "mask": tf.constant(1),
          "input": pp_builder.get_preprocess_fn(pp_fn)(example),
      }

    self.data = common.get_jax_process_dataset(
        dataset,
        split,
        global_batch_size=batch_size,
        add_tfds_id=True,
        pp_fn=preprocess)

  def run(self, params):
    """Run eval."""
    psnrs = []

    for batch in self.data.as_numpy_iterator():
      # Outputs is a dict with values shaped (gather/same, devices, batch, ...)
      out = self.predict_fn(params, batch)

      if jax.process_index():  # Host0 gets all preds and does eval.
        continue

      # First, we remove the "gather" dim and transfer the result to host,
      # leading to numpy arrays of (devices, device_batch, ...)
      out = jax.tree_map(lambda x: jax.device_get(x[0]), out)
      mask = out["mask"]
      batch_psnrs = out["psnr"][mask != 0]
      psnrs.extend(batch_psnrs)

    if jax.process_index():  # Host0 gets all preds and does eval.
      return

    yield "PSNR", np.mean(psnrs)


@functools.partial(jax.vmap, in_axes=[0, 0, None])
def _psnr(img0, img1, dynamic_range):
  mse = jnp.mean(jnp.power(img0 - img1, 2))
  return 20. * jnp.log10(dynamic_range) - 10. * jnp.log10(mse)
