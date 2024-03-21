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

"""Tests for the IRevNet adaptor."""

from big_vision.models.proj.givt import adaptor
import jax
from jax import random
import jax.numpy as jnp

from absl.testing import absltest


class AdaptorTest(googletest.TestCase):

  def test_inversion(self):
    num_channels = 8
    input_shape = (1, 24, 24, num_channels)

    rng = random.PRNGKey(758493)
    _, inp_rng, init_rng, data_rng = jax.random.split(rng, 4)

    dummy_x = random.normal(inp_rng, shape=input_shape)
    real_x = jax.random.normal(data_rng, shape=input_shape)

    model = adaptor.IRevNet(
        num_blocks=4,
        num_channels=num_channels,
        dropout_rate=0.0,
    )
    params = model.init(init_rng, dummy_x)

    real_y = model.apply(params, real_x, method=model.forward)
    real_x_ = model.apply(params, real_y, method=model.inverse)
    self.assertTrue(jnp.allclose(real_x, real_x_, atol=1e-5))


if __name__ == "__main__":
  googletest.main()
