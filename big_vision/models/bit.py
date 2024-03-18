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

"""ResNet V1 with GroupNorm."""

from typing import Optional, Sequence, Union

from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.numpy as jnp
import numpy as np


def weight_standardize(w, axis, eps):
  w = w - jnp.mean(w, axis=axis)
  w = w / (jnp.std(w, axis=axis) + eps)
  return w


class StdConv(nn.Conv):

  def param(self, name, *a, **kw):
    param = super().param(name, *a, **kw)
    if name == "kernel":
      param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
    return param


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""
  nmid: Optional[int] = None
  strides: Sequence[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    nmid = self.nmid or x.shape[-1] // 4
    nout = nmid * 4

    residual = x
    if x.shape[-1] != nout or self.strides != (1, 1):
      residual = StdConv(nout, (1, 1), self.strides, use_bias=False,
                         name="conv_proj")(residual)
      residual = nn.GroupNorm(name="gn_proj")(residual)

    y = StdConv(nmid, (1, 1), use_bias=False, name="conv1")(x)
    y = nn.GroupNorm(name="gn1")(y)
    y = nn.relu(y)
    y = StdConv(nmid, (3, 3), self.strides, use_bias=False, name="conv2")(y)
    y = nn.GroupNorm(name="gn2")(y)
    y = nn.relu(y)
    y = StdConv(nout, (1, 1), use_bias=False, name="conv3")(y)

    y = nn.GroupNorm(name="gn3", scale_init=nn.initializers.zeros)(y)
    y = nn.relu(residual + y)
    return y


class ResNetStage(nn.Module):
  """One stage of ResNet."""
  block_size: int
  first_stride: Sequence[int] = (1, 1)
  nmid: Optional[int] = None

  @nn.compact
  def __call__(self, x):
    x = ResidualUnit(self.nmid, strides=self.first_stride, name="unit1")(x)
    for i in range(1, self.block_size):
      x = ResidualUnit(self.nmid, name=f"unit{i + 1}")(x)
    return x


class Model(nn.Module):
  """ResNetV1."""
  num_classes: Optional[int] = None
  width: float = 1
  depth: Union[int, Sequence[int]] = 50

  @nn.compact
  def __call__(self, image, *, train=False):
    del train  # Unused
    blocks = get_block_desc(self.depth)
    width = int(64 * self.width)

    out = {}

    # Root block
    x = StdConv(width, (7, 7), (2, 2), use_bias=False, name="conv_root")(image)
    x = nn.GroupNorm(name="gn_root")(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
    out["stem"] = x

    # Stages
    x = ResNetStage(blocks[0], nmid=width, name="block1")(x)
    out["stage1"] = x
    for i, block_size in enumerate(blocks[1:], 1):
      x = ResNetStage(block_size, nmid=width * 2 ** i,
                      first_stride=(2, 2), name=f"block{i + 1}")(x)
      out[f"stage{i + 1}"] = x
    out["pre_logits_2d"] = x

    # Head
    x = out["pre_logits"] = jnp.mean(x, axis=(1, 2))

    if self.num_classes:
      head = nn.Dense(self.num_classes, name="head",
                      kernel_init=nn.initializers.zeros)
      out["logits_2d"] = head(out["pre_logits_2d"])
      x = out["logits"] = head(out["pre_logits"])

    return x, out


# A dictionary mapping the number of layers in a resnet to the number of
# blocks in each stage of the model.
# NOTE: Does not include 18/34 as they also need non-bottleneck block!
def get_block_desc(depth):
  if isinstance(depth, list):  # Be robust to silly mistakes.
    depth = tuple(depth)
  return {
      26: [2, 2, 2, 2],  # From timm, gets ~75% on ImageNet.
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }.get(depth, depth)


def fix_old_checkpoints(params):
  """Modifies params from old checkpoints to run with current implementation."""
  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))
  # Old linen used to store non-squeezed GN params.
  params = flax.traverse_util.unflatten_dict({
      k: np.squeeze(v) if (set(k)
                           & {"gn_root", "gn_proj", "gn1", "gn2", "gn3"}) else v
      for k, v in flax.traverse_util.flatten_dict(params).items()
  })
  return params


def load(init_params, init_file, model_cfg, dont_load=()):
  """Load init from checkpoint."""
  del model_cfg  # Unused
  params = utils.load_params(init_file)
  params = common.merge_params(params, init_params, dont_load)
  params = fix_old_checkpoints(params)
  return params
