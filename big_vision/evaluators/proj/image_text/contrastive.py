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

"""Evaluator for the contrastive task.

DON'T COMPARE ACROSS RUNS, use for training health monitoring only.

Note that this evaluator's `ncorrect_minibatch` is only a rough proxy for
training progress and does not report the actual `ncorrect`: when the same
labels found multiple times in a batch, then the reported value is biased
towards lower values.

Also note that the `ncorrect_minibatch` is a function of batch size (it's a lot
easier to find correct values in small batches).
"""
import functools

from big_vision import input_pipeline
import big_vision.datasets.core as ds_core
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import jax
import jax.numpy as jnp
import numpy as np


def _all_gather(z):
  """All gather and flatten first two dims."""
  gather_flat = lambda x: jnp.concatenate(jax.lax.all_gather(x, "batch"), 0)
  return jax.tree_map(gather_flat, z)


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@functools.lru_cache(None)
def get_eval_fn(predict_fn, use_global_batch):
  """Produces eval function, also applies pmap."""

  @functools.partial(jax.pmap, axis_name="batch")
  def _eval_fn(params, images, labels, mask):
    zimg, ztxt, extras = predict_fn(params, images, labels)

    if use_global_batch:
      zimg, ztxt, mask = _all_gather((zimg, ztxt, mask))

    # Temperature won't affect ranking for accuracy, but impacts loss magnitude.
    losses, measurements = u.bidirectional_contrastive_loss(
        zimg, ztxt, extras["t"], mask, reduction=False)
    l = jax.lax.psum(losses * mask, axis_name="batch")
    c = jax.lax.psum(measurements["ncorrect"] * mask, axis_name="batch")
    n = jax.lax.psum(mask, axis_name="batch")
    return c, l, n

  return _eval_fn


class Evaluator:
  """Contrastive evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size,
               use_global_batch, cache_final=True,
               cache_raw=False, prefetch=1, label_key="labels"):
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), pp_fn, batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_input_pipeline(self.ds, prefetch)
    self.eval_fn = get_eval_fn(predict_fn, use_global_batch)
    self.label_key = label_key

  def run(self, params):
    """Computes all metrics."""
    l, c, nseen = 0, 0, 0
    for _, batch in zip(range(self.steps), self.data_iter):
      labels, mask = batch.pop(self.label_key), batch.pop("_mask")
      batch_ncorrect, batch_losses, batch_n = self.eval_fn(
          params, batch["image"], labels, mask)
      # All results are a replicated array shaped as follows:
      # (local_devices, per_device_batch_size, elem_shape...)
      # with each local device's entry being identical as they got psum'd.
      # So let's just take the first one to the host as numpy.
      c += np.sum(np.array(batch_ncorrect[0]))
      l += np.sum(np.array(batch_losses[0]))
      nseen += np.sum(np.array(batch_n[0]))
    yield ("ncorrect_minibatch", c / nseen)
    yield ("loss", l / nseen)
