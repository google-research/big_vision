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

"""BiT models as in the paper (ResNet V2) w/ loading of public weights.

See reproduction proof: http://(internal link)/qY70qs6j944
"""

import functools
import re
from typing import Optional, Sequence, Union

from big_vision import utils as u
from big_vision.models import bit
from big_vision.models import common
import flax.linen as nn
import jax.numpy as jnp


def standardize(x, axis, eps):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x


# Defined our own, because we compute normalizing variance slightly differently,
# which does affect performance when loading pre-trained weights!
class GroupNorm(nn.Module):
  """Group normalization (arxiv.org/abs/1803.08494)."""
  ngroups: int = 32

  @nn.compact
  def __call__(self, x):

    input_shape = x.shape
    group_shape = x.shape[:-1] + (self.ngroups, x.shape[-1] // self.ngroups)

    x = x.reshape(group_shape)

    # Standardize along spatial and group dimensions
    x = standardize(x, axis=[1, 2, 4], eps=1e-5)
    x = x.reshape(input_shape)

    bias_scale_shape = tuple([1, 1, 1] + [input_shape[-1]])
    x = x * self.param('scale', nn.initializers.ones, bias_scale_shape)
    x = x + self.param('bias', nn.initializers.zeros, bias_scale_shape)
    return x


class StdConv(nn.Conv):

  def param(self, name, *a, **kw):
    param = super().param(name, *a, **kw)
    if name == 'kernel':
      param = standardize(param, axis=[0, 1, 2], eps=1e-10)
    return param


class RootBlock(nn.Module):
  """Root block of ResNet."""
  width: int

  @nn.compact
  def __call__(self, x):
    x = StdConv(self.width, (7, 7), (2, 2), padding=[(3, 3), (3, 3)],
                use_bias=False, name='conv_root')(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
    return x


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""
  nmid: Optional[int] = None
  strides: Sequence[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    nmid = self.nmid or x.shape[-1] // 4
    nout = nmid * 4
    conv = functools.partial(StdConv, use_bias=False)

    residual = x
    x = GroupNorm(name='gn1')(x)
    x = nn.relu(x)

    if x.shape[-1] != nout or self.strides != (1, 1):
      residual = conv(nout, (1, 1), self.strides, name='conv_proj')(x)

    x = conv(nmid, (1, 1), name='conv1')(x)
    x = GroupNorm(name='gn2')(x)
    x = nn.relu(x)
    x = conv(nmid, (3, 3), self.strides, padding=[(1, 1), (1, 1)],
             name='conv2')(x)
    x = GroupNorm(name='gn3')(x)
    x = nn.relu(x)
    x = conv(nout, (1, 1), name='conv3')(x)

    return x + residual


class ResNetStage(nn.Module):
  """A stage (sequence of same-resolution blocks)."""
  block_size: int
  nmid: Optional[int] = None
  first_stride: Sequence[int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    out = {}
    x = out['unit01'] = ResidualUnit(
        self.nmid, strides=self.first_stride, name='unit01')(x)
    for i in range(1, self.block_size):
      x = out[f'unit{i+1:02d}'] = ResidualUnit(
          self.nmid, name=f'unit{i+1:02d}')(x)
    return x, out


class Model(nn.Module):
  """ResNetV2."""
  num_classes: Optional[int] = None
  width: int = 1
  depth: Union[int, Sequence[int]] = 50  # 50/101/152, or list of block depths.
  head_zeroinit: bool = True

  @nn.compact
  def __call__(self, image, *, train=False):
    blocks = bit.get_block_desc(self.depth)
    width = int(64 * self.width)
    out = {}

    x = out['stem'] = RootBlock(width=width, name='root_block')(image)

    # Blocks
    x, out['stage1'] = ResNetStage(blocks[0], nmid=width, name='block1')(x)
    for i, block_size in enumerate(blocks[1:], 1):
      x, out[f'stage{i + 1}'] = ResNetStage(
          block_size, width * 2 ** i,
          first_stride=(2, 2), name=f'block{i + 1}')(x)

    # Pre-head
    x = out['norm_pre_head'] = GroupNorm(name='norm-pre-head')(x)
    x = out['pre_logits_2d'] = nn.relu(x)
    x = out['pre_logits'] = jnp.mean(x, axis=(1, 2))

    # Head
    if self.num_classes:
      kw = {'kernel_init': nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name='head', **kw)
      out['logits_2d'] = head(out['pre_logits_2d'])
      x = out['logits'] = head(out['pre_logits'])

    return x, out


def load(init_params, init_file, model_cfg, dont_load=()):
  """Loads the TF-dumped NumPy or big_vision checkpoint.

  Args:
    init_params: random init params from which the new head is taken.
    init_file: comes from `config.model_init`, can either be an absolute
      path (ie starts with /) to the checkpoint, or a string like
      "L-imagenet2012" describing one of the variants from the paper.
    model_cfg: the model configuration.
    dont_load: list of param names to be reset to init.

  Returns:
    The loaded parameters.
  """

  # Support for vanity model names from the paper.
  vanity = {
      'FunMatch-224px-i1k82.8': 'gs://bit_models/distill/R50x1_224.npz',
      'FunMatch-160px-i1k80.5': 'gs://bit_models/distill/R50x1_160.npz',
  }
  if init_file[0] in ('L', 'M', 'S'):  # The models from the original paper.
    # Supported names are of the following type:
    # - 'M' or 'S': the original "upstream" model without fine-tuning.
    # - 'M-ILSVRC2012': i21k model fine-tuned on i1k.
    # - 'M-run0-caltech101': i21k model fine-tuned on VTAB's caltech101.
    #    each VTAB fine-tuning was run 3x, so there's run0, run1, run2.
    if '-' in init_file:
      up, down = init_file[0], init_file[1:]
    else:
      up, down = init_file, ''
    down = {'-imagenet2012': '-ILSVRC2012'}.get(down, down)  # normalize
    fname = f'BiT-{up}-R{model_cfg.depth}x{model_cfg.width}{down}.npz'
    fname = f'gs://bit_models/{fname}'
  else:
    fname = vanity.get(init_file, init_file)

  params = u.load_params(fname)
  params = maybe_convert_big_transfer_format(params)
  return common.merge_params(params, init_params, dont_load)


def maybe_convert_big_transfer_format(params_tf):
  """If the checkpoint comes from legacy codebase, convert it."""

  # Only do anything at all if we recognize the format.
  if 'resnet' not in params_tf:
    return params_tf

  # For ease of processing and backwards compatibility, flatten again:
  params_tf = dict(u.tree_flatten_with_names(params_tf)[0])

  # Works around some files containing weird naming of variables:
  for k in list(params_tf):
    k2 = re.sub('/standardized_conv2d_\\d+/', '/standardized_conv2d/', k)
    if k2 != k:
      params_tf[k2] = params_tf[k]
      del params_tf[k]

  params = {
      'root_block': {'conv_root': {'kernel': params_tf[
          'resnet/root_block/standardized_conv2d/kernel']}},
      'norm-pre-head': {
          'bias': params_tf['resnet/group_norm/beta'][None, None, None],
          'scale': params_tf['resnet/group_norm/gamma'][None, None, None],
      },
      'head': {
          'kernel': params_tf['resnet/head/conv2d/kernel'][0, 0],
          'bias': params_tf['resnet/head/conv2d/bias'],
      }
  }

  for block in ('block1', 'block2', 'block3', 'block4'):
    params[block] = {}
    units = set([re.findall(r'unit\d+', p)[0] for p in params_tf.keys()
                 if p.find(block) >= 0])
    for unit in units:
      params[block][unit] = {}
      for i, group in enumerate('abc', 1):
        params[block][unit][f'conv{i}'] = {
            'kernel': params_tf[f'resnet/{block}/{unit}/{group}/standardized_conv2d/kernel']  # pylint: disable=line-too-long
        }
        params[block][unit][f'gn{i}'] = {
            'bias': params_tf[f'resnet/{block}/{unit}/{group}/group_norm/beta'][None, None, None],  # pylint: disable=line-too-long
            'scale': params_tf[f'resnet/{block}/{unit}/{group}/group_norm/gamma'][None, None, None],  # pylint: disable=line-too-long
        }

      projs = [p for p in params_tf.keys()
               if p.find(f'{block}/{unit}/a/proj') >= 0]
      assert len(projs) <= 1
      if projs:
        params[block][unit]['conv_proj'] = {
            'kernel': params_tf[projs[0]]
        }

  return params
