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

"""Tests for vision-text-transformer."""
from absl.testing import absltest

from big_vision.models.proj.uvim import vtt
import jax
import jax.numpy as jnp
import ml_collections


class VTTTest(absltest.TestCase):

  def test_vtt_with_1_step(self):
    model_config = ml_collections.ConfigDict(dict(
        input_size=(224, 224),
        patches={"size": (16, 16)},
        num_heads=2,
        num_layers=2,
        mlp_dim=128,
        emb_dim=64,
        vocab_size=500))
    batch_size, max_len = 8, 50
    image = jnp.ones((batch_size, 224, 224, 3))
    text = jnp.ones((batch_size, max_len), dtype=jnp.int32)

    m = vtt.Model(**model_config)
    variables = m.init(jax.random.PRNGKey(42), image, text)
    self.assertCountEqual(variables.keys(), ["params"])

    params = variables["params"]
    out = m.apply({"params": params}, image, text)
    expected_shape = (batch_size, max_len, model_config.vocab_size)
    self.assertEqual(out.shape, expected_shape)


if __name__ == "__main__":
  absltest.main()
