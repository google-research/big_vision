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
from big_vision.models.proj.givt import givt
from big_vision.models.proj.givt import parallel_decode
import chex
import jax
import jax.numpy as jnp

from absl.testing import absltest


_BATCH_SIZE = 2
_OUT_DIM = 4
_SEQ_LEN = 6
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
      style="masked",
  )
  config.update(overwrites)
  return givt.Model(**config)


def _mask(*flags):
  return jnp.asarray(flags).astype(jnp.bool_)


class HelperTest(googletest.TestCase):

  def test_get_first_n(self):
    with self.subTest("ordered"):
      values = jnp.asarray([4, 3, 2, 1, 0])
      k = jnp.asarray([3], jnp.int32)
      chex.assert_trees_all_equal(
          parallel_decode._get_bottom_k_mask(values, k), _mask(0, 0, 1, 1, 1)
      )

    with self.subTest("equal_values"):
      values = jnp.ones((5,))
      k = jnp.asarray([3], jnp.int32)
      chex.assert_trees_all_equal(
          parallel_decode._get_bottom_k_mask(values, k), _mask(1, 1, 1, 0, 0)
      )

    with self.subTest("equal_values"):
      values = jnp.asarray([1, 2, 2, 2, 3])
      k = jnp.asarray([3], jnp.int32)
      chex.assert_trees_all_equal(
          parallel_decode._get_bottom_k_mask(values, k), _mask(1, 1, 1, 0, 0)
      )


class ParallelDecodeTest(parameterized.TestCase):

  def _make_model(self, **overwrites):
    model = _make_test_model(**overwrites)
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
        train=False,
    )
    return model, variables

  def _test_model(self, rng, model, variables, config):
    labels = jnp.ones((_BATCH_SIZE,), dtype=jnp.int32)
    state = parallel_decode.decode_masked(
        rng,
        seq_len=_SEQ_LEN,
        feature_dim=_OUT_DIM,
        labels=labels,
        model=model,
        variables=variables,
        config=config,
    )
    self.assertEqual(int(state.step), 4)
    # Each point uncovered exactly once.
    chex.assert_trees_all_equal(
        state.uncovered_per_step.sum(0),
        jnp.ones((_BATCH_SIZE, _SEQ_LEN), dtype=jnp.int32),
    )

  @parameterized.product(
      rng_seed=[1, 2],
      choice_temperature=[1.0, 4.0],
      multivariate=[True, False],
  )
  def test_decode_masked(self, rng_seed, choice_temperature, multivariate):
    rng = jax.random.PRNGKey(rng_seed)
    model, variables = self._make_model(
        num_mixtures=1 if multivariate else _NUM_MIXTURES,
        multivariate=multivariate,
    )
    config = parallel_decode.MaskedGenerationConfig(
        num_steps=4,
        choice_temperature=choice_temperature,
    )
    self._test_model(rng, model, variables, config)

  @parameterized.product(
      rng_seed=[1, 2],
      choice_temperature=[1.0, 4.0],
      w=[0.0, 1.0, 3.0],
      per_channel_mixtures=[True, False],
  )
  def test_cfg(self, rng_seed, choice_temperature, w, per_channel_mixtures):
    rng = jax.random.PRNGKey(rng_seed)
    model, variables = self._make_model(
        num_mixtures=1 if per_channel_mixtures else 3,
        drop_labels_probability=0.1,
        per_channel_mixtures=per_channel_mixtures,
    )
    config = parallel_decode.MaskedGenerationConfig(
        num_steps=4,
        choice_temperature=choice_temperature,
        cfg_inference_weight=w,
    )
    self._test_model(rng, model, variables, config)


if __name__ == "__main__":
  googletest.main()
