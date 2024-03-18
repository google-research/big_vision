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

"""MLP-Mixer model."""

from typing import Optional, Tuple
from absl import logging

from big_vision import utils
from big_vision.models import common

import einops
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp


class MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.mlp_dim)(x)
    y = nn.gelu(y)
    return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
  """Mixer block layer."""
  tokens_mlp_dim: int
  channels_mlp_dim: int
  drop_p: float

  @nn.compact
  def __call__(self, x, *, train=False):
    y = nn.LayerNorm()(x)
    y = jnp.swapaxes(y, 1, 2)
    y = MlpBlock(self.tokens_mlp_dim, name="token_mixing")(y)
    y = jnp.swapaxes(y, 1, 2)
    x = x + y * _stoch_depth_mask(x, self.drop_p, not train, self.make_rng)
    y = nn.LayerNorm()(x)
    y = MlpBlock(self.channels_mlp_dim, name="channel_mixing")(y)
    return x + y * _stoch_depth_mask(x, self.drop_p, not train, self.make_rng)


class MlpMixer(nn.Module):
  """Mixer architecture."""
  patch_size: Tuple[int, int]
  num_classes: Optional[int]
  num_blocks: int
  hidden_dim: int
  tokens_mlp_dim: int
  channels_mlp_dim: int
  model_name: Optional[str] = None
  stoch_depth: float = 0.0

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}
    x = out["stem"] = nn.Conv(self.hidden_dim, self.patch_size,
                              strides=self.patch_size, name="stem")(image)
    x = out["input_tokens"] = einops.rearrange(x, "n h w c -> n (h w) c")
    for i in range(self.num_blocks):
      drop_p = (i / max(self.num_blocks - 1, 1)) * self.stoch_depth
      x = out[f"block_{i}"] = MixerBlock(
          self.tokens_mlp_dim, self.channels_mlp_dim, drop_p)(x, train=train)
    x = nn.LayerNorm(name="pre_head_layer_norm")(x)
    x = out["pre_logits"] = jnp.mean(x, axis=1)
    if self.num_classes:
      x = out["logits"] = nn.Dense(
          self.num_classes, kernel_init=nn.initializers.zeros, name="head")(x)
    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function to easily create a Model variant like "L/16"."""

  if variant is not None:
    model_size, patch = variant.split("/")
    kw.setdefault("patch_size", (int(patch), int(patch)))
    config = {
        "S": {
            "hidden_dim": 512,
            "num_blocks": 8,
            "channels_mlp_dim": 2048,
            "tokens_mlp_dim": 256
        },
        "B": {
            "hidden_dim": 768,
            "num_blocks": 12,
            "channels_mlp_dim": 3072,
            "tokens_mlp_dim": 384
        },
        "L": {
            "hidden_dim": 1024,
            "num_blocks": 24,
            "channels_mlp_dim": 4096,
            "tokens_mlp_dim": 512
        },
        "H": {
            "hidden_dim": 1280,
            "num_blocks": 32,
            "channels_mlp_dim": 5120,
            "tokens_mlp_dim": 640
        },
    }[model_size]

    for k, v in config.items():
      kw.setdefault(k, v)

  logging.info("Mixer config: %s", kw)
  return MlpMixer(num_classes=num_classes, **kw)


def load(init_params, init_file, model_cfg, dont_load=()):
  """Load checkpoint."""

  del model_cfg
  # Shortcut names for some canonical paper checkpoints:
  init_file = {
      # pylint: disable=line-too-long
      # Pretrained models from the MLP-Mixer paper: https://arxiv.org/abs/2105.01601.
      "B-i1k/16": "gs://mixer_models/imagenet1k/Mixer-B_16.npz",
      "L-i1k/16": "gs://mixer_models/imagenet1k/Mixer-L_16.npz",
      "B-i21k/16": "gs://mixer_models/imagenet21k/Mixer-B_16.npz",
      "L-i21k/16": "gs://mixer_models/imagenet21k/Mixer-L_16.npz",
      # pylint: enable=line-too-long
  }.get(init_file, init_file)
  restored_params = utils.load_params(init_file)
  restored_params = flax.training.checkpoints.convert_pre_linen(restored_params)

  if "Mixer" in restored_params:
    restored_params["pre_head_layer_norm"] = restored_params["Mixer"].pop(
        "encoder_norm"
    )
    restored_params["stem"] = restored_params.pop("embedding")
    def unflatten_dense(d):
      return {
          "Dense_0": {
              "bias": d["bias1"].squeeze(),
              "kernel": d["kernel1"].squeeze(),
          },
          "Dense_1": {
              "bias": d["bias2"].squeeze(),
              "kernel": d["kernel2"].squeeze(),
          },
      }
    for k, v in restored_params["Mixer"].items():
      assert k.startswith("encoderblock_"), k
      v["token_mixing"] = unflatten_dense(v.pop("token_mixing_phase_0"))
      v["channel_mixing"] = unflatten_dense(v.pop("channel_mixing_phase_0"))
      restored_params["MixerBlock_" + k[len("encoderblock_"):]] = v
    del restored_params["Mixer"]

  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load)

  return restored_params


def _stoch_depth_mask(x, drop_p, deterministic, make_rng):
  if not deterministic and drop_p:
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return 1.0 - jax.random.bernoulli(make_rng("dropout"), drop_p, shape)
  return 1.0
