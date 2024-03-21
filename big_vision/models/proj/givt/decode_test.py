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

from absl.testing import parameterized
from big_vision.models.proj.givt import decode
from big_vision.models.proj.givt import givt
import jax
import jax.numpy as jnp

from absl.testing import absltest


_BATCH_SIZE = 2
_OUT_DIM = 4
_IMG_DIM = 8
_PATCH_SIZE = 2
_SEQ_LEN = _IMG_DIM // _PATCH_SIZE * _IMG_DIM // _PATCH_SIZE
_NUM_MIXTURES = 4


def _make_test_model(**overwrites):
  config = dict(
      num_heads=2,
      num_decoder_layers=1,
      mlp_dim=64,
      emb_dim=16,
      patches=(_PATCH_SIZE, _PATCH_SIZE),
      input_size=(_IMG_DIM, _IMG_DIM),
      seq_len=_SEQ_LEN,
      out_dim=_OUT_DIM,
      num_mixtures=_NUM_MIXTURES,
      style="ar",
  )
  config.update(overwrites)
  return givt.Model(**config)


class DecodeTest(parameterized.TestCase):

  def _make_model(self, **overwrites):
    model = _make_test_model(**overwrites)
    sequence = jax.random.uniform(
        jax.random.PRNGKey(0), (_BATCH_SIZE, _SEQ_LEN, _OUT_DIM)
    )
    labels = jax.random.uniform(
        jax.random.PRNGKey(0), (_BATCH_SIZE,), maxval=10
    ).astype(jnp.int32)
    variables = model.init(
        jax.random.PRNGKey(0),
        sequence,
        labels,
        train=False,
        image=jnp.zeros((_BATCH_SIZE, _IMG_DIM, _IMG_DIM, 3), dtype=jnp.float32)
        if model.has_encoder
        else None,
    )
    return model, variables

  def _test_model(self, rng, model, variables, config):
    labels = jnp.ones((_BATCH_SIZE,), dtype=jnp.int32)
    if model.has_encoder:
      cond_image = jnp.zeros(
          (_BATCH_SIZE, _IMG_DIM, _IMG_DIM, 3), dtype=jnp.float32
      )
    else:
      cond_image = None
    result, logprobs = decode.generate(
        params=variables,
        seed=rng,
        seq_len=_SEQ_LEN,
        feature_dim=_OUT_DIM,
        labels=labels,
        model=model,
        config=config,
        cond_image=cond_image,
    )
    # TODO: More expressive tests? Eg for causality, and caching.
    self.assertEqual(result.shape, (_BATCH_SIZE, _SEQ_LEN, _OUT_DIM))
    self.assertTrue(jnp.allclose(logprobs, jnp.zeros_like(logprobs), atol=1e-5))

  @parameterized.product(
      rng_seed=[1, 2],
      encoder=[True, False],
  )
  def test_simple(self, rng_seed, encoder):
    rng = jax.random.PRNGKey(rng_seed)
    model, variables = self._make_model(
        num_layers=1 if encoder else 0
    )
    assert model.has_encoder == encoder
    self._test_model(rng, model, variables, config={})

  @parameterized.product(
      rng_seed=[1, 2],
      cfg_inference_weight=[0.0, 1.0, 3.0],
      per_channel_mixtures=[True, False],
  )
  def test_cfg(self, rng_seed, cfg_inference_weight, per_channel_mixtures):
    rng = jax.random.PRNGKey(rng_seed)
    model, variables = self._make_model(
        num_mixtures=1 if per_channel_mixtures else 3,
        drop_labels_probability=0.1,
        per_channel_mixtures=per_channel_mixtures,
    )
    config = {"cfg_inference_weight": cfg_inference_weight}
    self._test_model(rng, model, variables, config)


if __name__ == "__main__":
  googletest.main()
