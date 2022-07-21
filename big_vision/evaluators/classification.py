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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import
from functools import partial, lru_cache

import big_vision.input_pipeline as input_pipeline
import big_vision.pp.builder as pp_builder
import big_vision.utils as u

import jax
import jax.numpy as jnp
import numpy as np


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@lru_cache(None)
def get_eval_fn(predict_fn, loss_name):
  """Produces eval function, also applies pmap."""
  @partial(jax.pmap, axis_name='batch')
  def _eval_fn(params, batch, labels, mask):
    logits, *_ = predict_fn(params, **batch)

    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)

    losses = getattr(u, loss_name)(
        logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(
        labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')
    return ncorrect, loss, n
  return _eval_fn


class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, dataset, split, pp_fn, batch_size, loss_name,
               data_dir=None, cache_final=True, cache_raw=False, prefetch=1,
               label_key='labels'):
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        dataset, split, pp_fn, batch_size, data_dir,
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_input_pipeline(self.ds, prefetch)
    self.eval_fn = get_eval_fn(predict_fn, loss_name)
    self.label_key = label_key

  def run(self, params):
    """Computes all metrics."""
    ncorrect, loss, nseen = 0, 0, 0
    for _, batch in zip(range(self.steps), self.data_iter):
      labels, mask = batch.pop(self.label_key), batch.pop('_mask')
      batch_ncorrect, batch_losses, batch_n = self.eval_fn(
          params, batch, labels, mask)
      # All results are a replicated array shaped as follows:
      # (local_devices, per_device_batch_size, elem_shape...)
      # with each local device's entry being identical as they got psum'd.
      # So let's just take the first one to the host as numpy.
      ncorrect += np.sum(np.array(batch_ncorrect[0]))
      loss += np.sum(np.array(batch_losses[0]))
      nseen += np.sum(np.array(batch_n[0]))
    yield ('prec@1', ncorrect / nseen)
    yield ('loss', loss / nseen)
