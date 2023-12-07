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

"""Evaluator for perplexity of a model."""
from big_vision.evaluators import mean
import big_vision.utils as u
import jax.numpy as jnp


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


def perplexity(predict_fn, normalize_by_seqlen):
  """Returns a function that computes perplexity."""

  def _perplexity_fn(train_state, batch, pad_token=0, **kw):
    logits, _ = predict_fn(train_state, batch, **kw)

    # Ignore perplexity on the padding label.
    weights = jnp.where(batch['labels'] != pad_token, 1, 0).astype(jnp.float32)
    if batch.get('label_masks') is not None:
      weights = weights * batch['label_masks']

    losses = u.weighted_softmax_xent(
        logits=logits, labels=batch['labels'],
        weights=weights, label_smoothing=0.0,
        reduction=False, normalize=normalize_by_seqlen)

    return {'perplexity': losses}
  return _perplexity_fn


class Evaluator(mean.Evaluator):
  """Perplexity evaluator."""

  def __init__(self, predict_fn, *a, normalize_by_seqlen=False, **kw):
    super().__init__(perplexity(predict_fn, normalize_by_seqlen), *a, **kw)
