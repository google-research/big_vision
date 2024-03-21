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

"""Tests for GIVT model."""

from absl.testing import parameterized
from big_vision.models.proj.givt import givt
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest


_BATCH_SIZE = 2
_OUT_DIM = 4
_SEQ_LEN = 16
_NUM_MIXTURES = 4


def _make_test_model(**overwrites):
  config = dict(
      num_heads=2,
      num_decoder_layers=1,
      mlp_dim=64,
      emb_dim=16,
      seq_len=_SEQ_LEN,
      out_dim=_OUT_DIM,
      num_mixtures=_NUM_MIXTURES,
  )
  config.update(overwrites)
  return givt.Model(**config)


class MaskedTransformerTest(parameterized.TestCase):

  @parameterized.product(rng_seed=[0])
  def test_masks(self, rng_seed):
    m = _make_test_model(style="masked")
    mask = m.get_input_mask_training(jax.random.PRNGKey(rng_seed), (2, 16))
    self.assertEqual(mask.shape, (2, 16))
    # At least one should definitly be masked out.
    self.assertTrue(np.all(mask.sum(-1) > 1))

  @parameterized.product(
      train=[True, False],
      multivariate=[True, False],
      per_channel_mixtures=[True, False],
      drop_labels_probability=[0.0, 0.1],
      style=["masked", "ar"],
  )
  def test_apply(
      self,
      train,
      multivariate,
      per_channel_mixtures,
      drop_labels_probability,
      style,
  ):
    if per_channel_mixtures and multivariate:
      self.skipTest("Not supported")
    model = _make_test_model(
        style=style,
        multivariate=multivariate,
        num_mixtures=1 if multivariate else _NUM_MIXTURES,
        per_channel_mixtures=per_channel_mixtures,
        drop_labels_probability=drop_labels_probability,
    )
    sequence = jax.random.uniform(
        jax.random.PRNGKey(0), (_BATCH_SIZE, _SEQ_LEN, _OUT_DIM)
    )
    labels = jax.random.uniform(
        jax.random.PRNGKey(0), (_BATCH_SIZE,), maxval=10
    ).astype(jnp.int32)
    input_mask = jax.random.uniform(
        jax.random.PRNGKey(0), (_BATCH_SIZE, _SEQ_LEN)
    ).astype(jnp.bool_)
    variables = model.init(
        jax.random.PRNGKey(0),
        sequence,
        labels,
        input_mask=input_mask,
        train=train,
    )
    logits, pdf = model.apply(
        variables, sequence, labels, input_mask=input_mask, train=train
    )
    nll = -pdf.log_prob(sequence)
    self.assertFalse(np.any(np.isnan(nll)))
    if multivariate:
      self.assertEqual(
          logits.shape, (_BATCH_SIZE, _SEQ_LEN, _OUT_DIM**2 + _OUT_DIM)
      )
      self.assertEqual(nll.shape, (_BATCH_SIZE, _SEQ_LEN))
    elif per_channel_mixtures:
      self.assertEqual(
          logits.shape,
          (_BATCH_SIZE, _SEQ_LEN, 3 * _NUM_MIXTURES * _OUT_DIM),
      )
      self.assertEqual(nll.shape, (_BATCH_SIZE, _SEQ_LEN, _OUT_DIM))
    else:
      self.assertEqual(
          logits.shape,
          (_BATCH_SIZE, _SEQ_LEN, _NUM_MIXTURES + _NUM_MIXTURES * _OUT_DIM * 2),
      )
      self.assertEqual(nll.shape, (_BATCH_SIZE, _SEQ_LEN))

    sample = pdf.sample(seed=jax.random.PRNGKey(0))
    self.assertEqual(sample.shape, (_BATCH_SIZE, _SEQ_LEN, _OUT_DIM))


if __name__ == "__main__":
  googletest.main()
