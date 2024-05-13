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

"""Evaluator for perplexity of a model."""
import functools

from big_vision.evaluators import mean
import big_vision.utils as u
import jax.numpy as jnp


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# Cache the function such that it won't always recompile (in mean evaluator).
@functools.cache
def perplexity(
    predict_fn, key='labels', shift_labels=True):
  """Returns a function that computes perplexity."""

  def _perplexity_fn(train_state, batch, **kw):
    logits, _ = predict_fn(train_state, batch, **kw)

    labels = batch[key]
    weights = batch.get('mask_loss', jnp.ones_like(labels))

    if shift_labels:
      labels = labels[:, 1:]
      weights = weights[:, 1:]

    losses = u.weighted_softmax_xent(
        logits=logits, labels=labels, weights=weights,
        reduction=False, normalize=False)
    normalizer = jnp.clip(weights.sum(axis=1), 2e-38)

    return {'sum': losses, 'avg': losses / normalizer}
  return _perplexity_fn


class Evaluator(mean.Evaluator):
  """Perplexity evaluator."""

  def __init__(self, predict_fn, *a, key='labels', shift_labels=False, **kw):
    kw.setdefault('prefetch', 0)  # More memory-saving default.
    super().__init__(perplexity(predict_fn, key, shift_labels), *a, **kw)
