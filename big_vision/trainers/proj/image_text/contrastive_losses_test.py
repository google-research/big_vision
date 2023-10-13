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

from absl.testing import absltest
from big_vision.trainers.proj.image_text import contrastive
import jax
import jax.numpy as jnp
import numpy as np


class ContrastiveTest(absltest.TestCase):

  def test_chunked_equivalence(self):
    d = jax.device_count()
    zimg = jax.random.normal(jax.random.PRNGKey(1), (d, 128, 512))
    ztxt = jax.random.uniform(jax.random.PRNGKey(2), (d, 128, 512))
    temperature = jnp.asarray(10.0, jnp.float32)
    bias = jnp.asarray(-24, jnp.float32)
    zimg = jax.device_put_sharded(list(zimg), jax.devices())
    ztxt = jax.device_put_sharded(list(ztxt), jax.devices())
    temperature = jax.device_put_replicated(temperature, jax.devices())
    bias = jax.device_put_replicated(bias, jax.devices())

    def compute_loss(fn):
      def _fn(*args):
        loss, _ = fn(*args)
        return jax.lax.pmean(loss, "batch")
      return np.asarray(jax.pmap(_fn, "batch")(zimg, ztxt, temperature, bias))

    np.testing.assert_array_almost_equal(
        compute_loss(contrastive.sigmoid_loss),
        compute_loss(contrastive.chunked_sigmoid_loss),
        decimal=5)


if __name__ == "__main__":
  absltest.main()
