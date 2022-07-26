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

"""Tests for vit vqvae model."""
from absl.testing import absltest

from big_vision.models.proj.uvim import vit
import jax
import jax.numpy as jnp
import ml_collections


class ViTVQVAEModelTest(absltest.TestCase):

  def test_model(self):
    model_config = ml_collections.ConfigDict({
        "input_size": (32, 32),
        "code_len": 4,
        "width": 16,
        "mlp_dim": 64,
        "num_heads": 4,
        "enc_depth": 1,
        "dec_depth": 1,
        "with_encoder_ctx": True,
        "with_decoder_ctx": True,
        "statistics_axis_name": None,
        "inputs": {
            "in1": (10, 3),
            "in2": (25,),
        },
        "outputs": {
            "out1": (5,),
            "out2": (20,),
        },
    })

    model = vit.Model(**model_config)
    batch_size = 4
    seq_len = (32 // 8) ** 2
    x = {
        "in1": jnp.zeros((batch_size, seq_len, 10, 3)),
        "in2": jnp.zeros((batch_size, seq_len, 25)),
    }
    ctx_image = jnp.zeros((batch_size,) + model_config.input_size + (3,))
    init_rngs = {
        "params": jax.random.PRNGKey(0),
        "state": jax.random.PRNGKey(1),
    }
    params = model.init(init_rngs, x, ctx=ctx_image)
    self.assertEqual(params.keys(), set(["params", "state"]))

    apply_rngs = {
        "dropout": jax.random.PRNGKey(0),
        "vqvae": jax.random.PRNGKey(0),
    }
    (logits, _), params = model.apply(
        params, x, ctx=ctx_image, train=True, update_dict=True,
        rngs=apply_rngs, mutable=["state"])
    self.assertEqual(logits.keys(), set(["out1", "out2"]))
    self.assertEqual(logits["out1"].shape, (batch_size, seq_len, 5))
    self.assertEqual(logits["out2"].shape, (batch_size, seq_len, 20))


if __name__ == "__main__":
  absltest.main()
