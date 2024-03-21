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

"""Simple VAE fork of the UViM VQ-VAE (proj/uvim/vit.py) with small changes."""

from typing import Optional, Sequence, Mapping, Any

from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
from big_vision.models.proj.givt import vae

import einops
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np


class Model(vae.Model):
  """ViT model."""

  input_size: Sequence[int] = (256, 256)
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  enc_depth: int = 6
  dec_depth: int = 6
  mlp_dim: Optional[int] = None
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  dropout: float = 0.0
  head_zeroinit: bool = True
  bottleneck_resize: bool = False
  inout_specs: Optional[Mapping[str, tuple[int, int]]] = None
  scan: bool = False
  remat_policy: str = "nothing_saveable"

  def setup(self) -> None:
    self.grid_size = np.array(self.input_size) // np.array(self.patch_size)

    self.embedding = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        padding="VALID", name="embedding")

    self.pos_embedding_encoder = vit.get_posemb(
        self, self.posemb, self.grid_size, self.width, "pos_embedding_encoder")
    self.encoder = vit.Encoder(
        depth=self.enc_depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        name="encoder")

    if not self.bottleneck_resize:
      self.bottleneck_downsample = self.param(
          "bottleneck_downsample",
          nn.initializers.xavier_uniform(),
          (np.prod(self.grid_size), self.code_len))

    if not self.bottleneck_resize:
      self.bottleneck_upsample = self.param(
          "bottleneck_upsample",
          nn.initializers.xavier_uniform(),
          (self.code_len, np.prod(self.grid_size)))

    self.pos_embedding_decoder = vit.get_posemb(
        self, self.posemb, self.grid_size, self.width, "pos_embedding_decoder")
    self.decoder = vit.Encoder(
        depth=self.dec_depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        name="decoder")

    # Setting num_outputs to 2 * codeword_dim to predict mean and variance per
    # element
    self.encoder_head = nn.Dense(self.codeword_dim * 2 or self.width * 2)
    self.decoder_stem = nn.Dense(self.width)

    kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}

    if self.inout_specs is not None:
      num_out_channels = sum(
          num_classes for _, num_classes in self.inout_specs.values())
    else:
      num_out_channels = 3

    self.head = nn.Dense(
        num_out_channels * np.prod(self.patch_size),
        name="decoder_head", **kw)

  def encode(
      self,
      x: jax.Array,
      *,
      train: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    if self.inout_specs is not None:
      one_hot_inputs = []
      for in_ch, num_classes in self.inout_specs.values():
        one_hot_inputs.append(nn.one_hot(x[..., in_ch], num_classes))
      x = jnp.concatenate(one_hot_inputs, axis=-1)
    x = self.embedding(x)
    x = einops.rearrange(x, "b h w c -> b (h w) c")

    x, _ = self.encoder(x + self.pos_embedding_encoder, deterministic=not train)

    if self.bottleneck_resize:
      x = einops.rearrange(x, "b (h w) c -> b h w c",
                           h=self.grid_size[0], w=self.grid_size[1])
      l = int(np.round(self.code_len ** 0.5))
      x = jax.image.resize(
          x, (x.shape[0], l, l, x.shape[3]),
          method="linear")
      x = einops.rearrange(x, "b h w c -> b (h w) c")
    else:
      x = jnp.einsum("btc,tn->bnc", x, self.bottleneck_downsample)

    x = self.encoder_head(x)

    mu, logvar = jnp.split(x, 2, axis=-1)
    return mu, logvar

  def decode(
      self,
      x: jax.Array,
      train: bool = False,
  ) -> jax.Array | Mapping[str, jax.Array]:
    x = self.decoder_stem(x)

    if self.bottleneck_resize:
      l = int(np.round(self.code_len ** 0.5))
      x = einops.rearrange(x, "b (h w) c -> b h w c", h=l, w=l)
      x = jax.image.resize(
          x, (x.shape[0], self.grid_size[0], self.grid_size[1], x.shape[3]),
          method="linear")
      x = einops.rearrange(x, "b h w c -> b (h w) c")
    else:
      x = jnp.einsum("bnc,nt->btc", x, self.bottleneck_upsample)

    x, _ = self.decoder(x + self.pos_embedding_decoder, deterministic=not train)
    x = self.head(x)
    # c = 3 for RGB images
    x = einops.rearrange(x, "b (h w) (p q c) -> b (h p) (w q) c",
                         h=self.grid_size[0], w=self.grid_size[1],
                         p=self.patch_size[0], q=self.patch_size[1])

    if self.inout_specs is None:
      x = jnp.clip(x, -1.0, 1.0)
    else:
      x_dict = {}
      channel_index = 0
      for name, (_, num_channels) in self.inout_specs.items():
        x_dict[name] = x[..., channel_index : channel_index + num_channels]
        channel_index += num_channels
      x = x_dict

    return x


def load(
    init_params: Any,
    init_file: str,
    model_params: Any = None,
    dont_load: Sequence[str] = (),
) -> Any:
  """Loads params from init checkpoint and merges into init_params."""
  del model_params
  params = flax.core.unfreeze(utils.load_params(init_file))
  if init_params is not None:
    params = common.merge_params(params, init_params, dont_load)
  return params
