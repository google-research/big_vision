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

"""Invertible adaptor based on iRevNet.

Based on the PyTorch version from:
https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/iRevNet.py
"""

from typing import Any, Optional, Sequence

from big_vision import utils
from big_vision.models import common
from big_vision.models.proj.givt import cnn
import einops
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp


def _split(x: jax.Array) -> tuple[jax.Array, jax.Array]:
  n = x.shape[-1] // 2
  x1 = x[:, :, :, :n]
  x2 = x[:, :, :, n:]
  return x1, x2


def _merge(x1: jax.Array, x2: jax.Array) -> jax.Array:
  return jnp.concatenate((x1, x2), axis=-1)


class IRevNetBlock(nn.Module):
  """iRevNet Block."""
  first: int = False
  dropout_rate: float = 0.
  num_channels: int = 2
  num_channels_bottleneck: Optional[int] = None
  num_grps_norm: int = 32

  @nn.compact
  def _fx2(self, x: jax.Array, train: bool = True) -> jax.Array:
    if not self.first:
      y = nn.GroupNorm(num_groups=self.num_grps_norm, name="gn_0")(x)
      y = nn.relu(y)
    else:
      y = x

    ks = (3, 3)     # hardcode kernel-size 3 for now
    y = nn.Conv(self.num_channels_bottleneck or self.num_channels,
                kernel_size=ks, padding=1, use_bias=False)(y)
    y = nn.GroupNorm(num_groups=self.num_grps_norm, name="gn_1")(y)
    y = nn.relu(y)

    y = nn.Conv(self.num_channels_bottleneck or self.num_channels,
                kernel_size=ks, padding=1, use_bias=False)(y)
    y = nn.Dropout(rate=self.dropout_rate, deterministic=(not train))(y)
    y = nn.GroupNorm(num_groups=self.num_grps_norm, name="gn_2")(y)
    y = nn.relu(y)

    y = nn.Conv(self.num_channels, kernel_size=ks, padding=1, use_bias=False)(y)

    return y

  def forward(
      self,
      x: tuple[jax.Array, jax.Array],
      train: bool = True,
  ) -> tuple[jax.Array, jax.Array]:
    """Bijective block forward."""
    x1, x2 = x[0], x[1]
    fx2 = self._fx2(x2, train=train)
    y1 = fx2 + x1
    return (x2, y1)

  def inverse(self,
              x: tuple[jax.Array, jax.Array],
              train: bool = True
              ) -> tuple[jax.Array, jax.Array]:
    """Bijective block inverse."""
    x2, y1 = x[0], x[1]
    fx2 = -self._fx2(x2, train=train)
    x1 = fx2 + y1
    return (x1, x2)


class IRevNet(nn.Module):
  """iRevNet."""
  num_blocks: int = 4
  num_channels: int = 4
  num_channels_bottleneck: Optional[int] = None
  dropout_rate: float = 0.0

  def setup(self) -> None:
    num_grps_norm = min(32, self.num_channels // 2)
    self.modules = [
        IRevNetBlock(
            first=(i == 0),
            num_channels=self.num_channels // 2,
            num_channels_bottleneck=(
                self.num_channels_bottleneck or self.num_channels) // 2,
            num_grps_norm=num_grps_norm,
            dropout_rate=self.dropout_rate,
        )
        for i in range(self.num_blocks)
    ]

  def forward(self, x: jax.Array, train: bool = True) -> jax.Array:
    out = _split(x)
    for m in self.modules:
      out = m.forward(out, train=train)
    out_bij = _merge(out[0], out[1])
    return out_bij

  def inverse(self, out_bij: jax.Array, train: bool = True) -> jax.Array:
    out = _split(out_bij)
    for m in reversed(self.modules):
      out = m.inverse(out, train=train)
    out = _merge(out[0], out[1])
    return out

  def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
    return self.forward(x, train=train)


class Model(IRevNet):
  """Wrapper for IRevNet to function as an adaptor in our setup."""

  pixel_shuffle_patch_size: tuple[int, int] = (1, 1)

  def forward(self, x: jax.Array, train: bool = True) -> jax.Array:
    # (b, code_len, ch) --> (b, h, w, ch) --> (b, code_len, ch)
    # h, w are the spatial dimensions after space-to-depth transformation
    h, w = cnn.get_h_w_pixelshuffle(x.shape[1], self.pixel_shuffle_patch_size)
    x = einops.rearrange(x, "b (h w) c -> b h w c", h=h, w=w)
    x = super().forward(x, train)
    x = einops.rearrange(x, "b h w c -> b (h w) c")   # (b, codelen, codeword_d)

    return x

  def inverse(self, out_bij: jax.Array, train: bool = True) -> jax.Array:
    # (b, code_len, ch) --> (b, h, w, ch) --> (b, code_len, ch)
    h, w = cnn.get_h_w_pixelshuffle(
        out_bij.shape[1], self.pixel_shuffle_patch_size)
    out_bij = einops.rearrange(out_bij, "b (h w) c -> b h w c", h=h, w=w)
    out_bij = super().inverse(out_bij, train)
    out_bij = einops.rearrange(out_bij, "b h w c -> b (h w) c")

    return out_bij


def load(
    init_params: Any,
    init_file: str,
    model_params: Any = None,
    dont_load: Sequence[str] = (),
) -> Any:
  """Loads params from init checkpoint and merges into init_params."""
  del model_params
  ckpt_params = flax.core.unfreeze(utils.load_params(init_file))
  if init_params is not None:
    ckpt_params = common.merge_params(ckpt_params, init_params, dont_load)
  return ckpt_params
