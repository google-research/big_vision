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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import

import functools

from big_vision.evaluators import common
import big_vision.utils as u
import jax
import jax.numpy as jnp


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@functools.cache
def get_eval_fn(predict_fn, loss_name):
  """Produces eval function, also applies pmap."""
  @jax.jit
  def _eval_fn(train_state, batch, labels, mask):
    logits, *_ = predict_fn(train_state, batch)

    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)

    loss = getattr(u, loss_name)(
        logits=logits, labels=labels, reduction=False)
    loss = jnp.sum(loss * mask)

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(
        labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jnp.sum(top1_correct * mask)
    nseen = jnp.sum(mask)
    return ncorrect, loss, nseen
  return _eval_fn


class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, loss_name, label_key='labels', **kw):
    self.get_data_iter, self.steps = common.eval_input_pipeline(**kw)
    self.eval_fn = get_eval_fn(predict_fn, loss_name)
    self.label_key = label_key

  def run(self, train_state):
    """Computes all metrics."""
    ncorrect, loss, nseen = 0, 0, 0
    for _, batch in zip(range(self.steps), self.get_data_iter()):
      labels, mask = batch.pop(self.label_key), batch.pop('_mask')
      batch_ncorrect, batch_losses, batch_nseen = jax.device_get(
          self.eval_fn(train_state, batch, labels, mask))
      ncorrect += batch_ncorrect
      loss += batch_losses
      nseen += batch_nseen
    yield ('prec@1', ncorrect / nseen)
    yield ('loss', loss / nseen)
