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

"""Abstract VAE model class.

Gaussian encoder and decoder (the latter assumed to have constant variance).

Inspiration drawn from https://github.com/pytorch/examples/tree/main/vae.
"""

import abc
from typing import Optional, Mapping


import flax.linen as nn
import jax
import jax.numpy as jnp


class Model(nn.Module, metaclass=abc.ABCMeta):
  """Abstract VAE model class."""

  codeword_dim: Optional[int] = None
  code_len: int = 256
  code_dropout: str = "none"

  @abc.abstractmethod
  def encode(
      self,
      x: jax.Array,
      *,
      train: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    ...

  def reparametrize(
      self,
      mu: jax.Array,
      logvar: jax.Array,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    std = jnp.exp(0.5 * logvar)
    if rng is None:
      rng = self.make_rng("dropout")
    eps = jax.random.normal(rng, shape=std.shape, dtype=std.dtype)
    return mu + std * eps

  @abc.abstractmethod
  def decode(
      self, x: jax.Array,
      train: bool = False,
  ) -> jax.Array | Mapping[str, jax.Array]:
    ...

  def code_dropout_fn(self, z: jax.Array, *, train: bool = False) -> jax.Array:
    # "seq" drops out tokens later in the sequence with higher probablility than
    # tokens earlier in the sequence.
    assert self.code_dropout in ["none", "seq", "random"]
    if train and self.code_dropout != "none":
      importance = jnp.linspace(1.0, 0.0, self.code_len + 2)[1:-1]
      thr = jax.random.uniform(self.make_rng("dropout"), z.shape[:1])
      mask = importance[None, :] > thr[:, None]
      if self.code_dropout == "random":
        mask = jax.random.permutation(
            self.make_rng("dropout"), mask, axis=-1, independent=True)
      z = z * mask[:, :, None]
    return z

  def __call__(
      self,
      x: jax.Array,
      *,
      train: bool = False,
  ) -> tuple[jax.Array | Mapping[str, jax.Array], Mapping[str, jax.Array]]:
    mu, logvar = self.encode(x, train=train)
    # Only reparametrize when training for simplicity.
    if train:
      z = self.reparametrize(mu, logvar)
    else:
      z = mu
    z = self.code_dropout_fn(z, train=train)
    x = self.decode(z, train=train)
    return x, {"mu": mu, "logvar": logvar, "z": z}
