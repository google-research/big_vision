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

"""Jet: A Modern Transformer-Based Normalizing Flow.

https://arxiv.org/abs/2412.15129
"""

import itertools
from typing import Any, Sequence

from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
import einops
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class DNN(nn.Module):
  """Main non-invertible compute block with a ViT used in coupling layers."""

  depth: int = 1
  emb_dim: int = 256
  num_heads: int = 4

  @nn.compact
  def __call__(self, x, context=None):
    out_dim = x.shape[-1]  # pytype: disable=attribute-error
    x = nn.Dense(self.emb_dim, name="init_proj")(x)
    posemb = self.param("posemb",
                        nn.initializers.normal(stddev=1/np.sqrt(self.emb_dim)),
                        (1,) + x.shape[1:], jnp.float32)
    x = x + posemb

    if context is not None:
      y = nn.MultiHeadDotProductAttention(
          num_heads=self.num_heads,
          qkv_features=self.emb_dim,
          out_kernel_init=nn.initializers.zeros,
          out_features=x.shape[-1])(x, context)
      x = x + y

    x, _ = vit.Encoder(depth=self.depth, num_heads=self.num_heads,
                       name="vit")(x)
    bias, scale = jnp.split(
        nn.Dense(2 * out_dim,
                 kernel_init=nn.initializers.zeros,
                 name="final_proj")(x),
        2, axis=-1)
    return bias, scale


class Coupling(nn.Module):
  """Coupling layer.

  Supports two kinds of couplings: channels-wise and spatial.

  The implementation is a bit convoluted, but it is necessary to support
  scan over layers. The coupling needs to be modulated by the input arguments
  and can not be modulated by the static attributes (because scan loops over
  a fixed Module instance).

  To make things worse, we want to support both channels-wise and spatial
  couplings, which need differently-shaped projections applied along different
  axes. Also they will have different output shapes. To work around this, we
  perform both projections, select the right one with jax.lax.cond, and then
  also do depth-to-space for spatial couplings to ensure consistent shapes.
  """

  depth: int = 1
  emb_dim: int = 256
  num_heads: int = 4
  scale_factor: float = 2.0

  @nn.compact
  def compact_setup(self, x, kind, channel_proj, spatial_proj, context=None):

    dnn = DNN(depth=self.depth, emb_dim=self.emb_dim,
              num_heads=self.num_heads, name="dnn")

    # Channel-wise split and merge utils.
    def split_channels(x):
      x = jnp.einsum("ntk,km->ntm", x, channel_proj, precision="highest")
      x1, x2 = jnp.split(x, 2, axis=-1)
      return x1, x2
    def merge_channels(x1, x2):
      x = jnp.concatenate([x1, x2], axis=-1)
      x = jnp.einsum("ntk,km->ntm", x, channel_proj.T, precision="highest")
      return x

    # Spatial split and merge utils. Note that we do extra reshape with the
    # the `cut` function to ensure that channel-wise and spatial-wise
    # projections result in the same output shape.
    def split_spatial(x):
      x = jnp.einsum("ntk,tm->nmk", x, spatial_proj, precision="highest")
      x1, x2 = jnp.split(x, 2, axis=-2)
      cut = lambda a: einops.rearrange(a, "... n (s c) -> ... (n s) c", s=2)
      x1, x2 = jax.tree.map(cut, (x1, x2))
      return x1, x2
    def merge_spatial(x1, x2):
      uncut = lambda a: einops.rearrange(a, "... (n s) c -> ... n (s c)", s=2)
      x1, x2 = jax.tree.map(uncut, (x1, x2))
      x = jnp.concatenate([x1, x2], axis=-2)
      x = jnp.einsum("ntk,tm->nmk", x, spatial_proj.T, precision="highest")
      return x

    # Dynamically select the right splitting
    x1, x2 = jax.lax.cond(kind, split_channels, split_spatial, x)

    # The same for forward/inverse, so compute inside the setup.
    bias, raw_scale = dnn(x1, context)
    scale = jax.nn.sigmoid(raw_scale) * self.scale_factor
    logdet = jax.nn.log_sigmoid(raw_scale) + jnp.log(self.scale_factor)
    logdet = jnp.sum(logdet, axis=range(1, logdet.ndim))

    return x1, x2, bias, scale, merge_channels, merge_spatial, logdet

  def forward(self, x, kind, channel_proj, spatial_proj, context=None):
    """Implements spatial (kind=0) or channel (kind=1) coupling."""
    x1, x2, bias, scale, merge_channels, merge_spatial, logdet = (
        self.compact_setup(x, kind, channel_proj, spatial_proj, context))
    x2 = (x2 + bias) * scale
    x = jax.lax.cond(kind, merge_channels, merge_spatial, x1, x2)
    return x, logdet

  def inverse(self, x, kind, channel_proj, spatial_proj, context=None):
    """Implements spatial (kind=0) or channel (kind=1) coupling."""
    x1, x2, bias, scale, merge_channels, merge_spatial, logdet = (
        self.compact_setup(x, kind, channel_proj, spatial_proj, context))
    x2 = (x2 / scale) - bias
    x = jax.lax.cond(kind, merge_channels, merge_spatial, x1, x2)
    return x, -logdet


class Model(nn.Module):
  """Jet: a normalizing flow model parameterized by ViT blocks."""

  depth: int = 2
  block_depth: int = 1
  emb_dim: int = 256
  num_heads: int = 4
  scale_factor: float = 2.0
  ps: int = 4
  channels_coupling_projs: Sequence[str] = ("random",)
  spatial_coupling_projs: Sequence[str] = ("checkerboard", "checkerboard-inv")
  kinds: Sequence[str] = ("channels", "channels", "spatial")

  @nn.compact
  def compact_setup(self, x):

    # `kinds` variable defines the sequence of channel and spatial couplings.
    # After that, we draw specific coupling projection types.
    def _interleave_couplings():
      kinds = itertools.cycle(self.kinds)
      cc = itertools.cycle(self.channels_coupling_projs)
      sc = itertools.cycle(self.spatial_coupling_projs)

      # Zero coupling is a placeholder and will never be executed. If it happens
      # due to a bug, the objective will explode.
      # Yields `(coupling_type, channel_proj type and spatial_proj type)`.
      while True:
        k = next(kinds)
        if k == "channels":
          yield 1, next(cc), "zero"
        elif k == "spatial":
          yield 0, "zero", next(sc)
        else:
          raise ValueError(f"Unknown coupling kind: {k}")

    kinds, c_proj_kinds, s_proj_kinds = (
        list(zip(*itertools.islice(_interleave_couplings(), self.depth))))
    kinds = jnp.array(kinds)

    # Initialize channel-wise coupling projections.
    channels_init_fn = get_channels_coupling_init(
        self.depth, x.shape[1:], self.ps, proj_kinds=c_proj_kinds)
    c_proj = self.param("channel_coupling_masks-FREEZE_ME", channels_init_fn,
                        jnp.float32)

    # Initialize spatial-wise coupling projections.
    spatial_init_fn = get_spatial_coupling_init(
        self.depth, x.shape[1:], self.ps, proj_kinds=s_proj_kinds)
    s_proj = self.param("spatial_coupling_masks-FREEZE_ME", spatial_init_fn,
                        jnp.float32)

    remat_coupling = nn.remat(
        Coupling,
        prevent_cse=False,
        policy=jax.checkpoint_policies.nothing_saveable,
        methods=("forward", "inverse",))
    block = remat_coupling(
        name="couplings",
        depth=self.block_depth,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        scale_factor=self.scale_factor,
    )

    def body_fn_forward(m, carry, kind, c_proj, s_proj, context):
      carry, y = m.forward(carry, kind, c_proj, s_proj, context)
      return carry, y

    def body_fn_inverse(m, carry, kind, c_proj, s_proj, context):
      carry, y = m.inverse(carry, kind, c_proj, s_proj, context)
      return carry, y

    scan_kwargs = dict(
        variable_axes={"params": 0},
        in_axes=(0, 0, 0, nn.broadcast),
        split_rngs={"params": True},
        length=self.depth)
    m_forward = nn.scan(body_fn_forward, **scan_kwargs)
    m_inverse = nn.scan(body_fn_inverse, **scan_kwargs, reverse=True)

    return block, m_forward, m_inverse, kinds, c_proj, s_proj

  def forward(self, x, context=None):

    block, m_forward, _, kinds, c_projs, s_projs = self.compact_setup(x)

    x = einops.rearrange(x, "b (h hp) (w wp) c -> b (h w) (hp wp c)",
                         hp=self.ps, wp=self.ps)

    x, logdet = m_forward(block, x, kinds, c_projs, s_projs, context)
    logdet = jnp.sum(logdet, axis=0)

    x = einops.rearrange(x, "b (h w) (hp wp c) -> b (h hp) (w wp) c",
                         hp=self.ps, wp=self.ps,
                         h=np.round(x.shape[1] ** 0.5).astype(np.int32))

    return x, logdet

  def inverse(self, x, context=None):

    block, _, m_inverse, kinds, c_projs, s_projs = self.compact_setup(x)

    x = einops.rearrange(x, "b (h hp) (w wp) c -> b (h w) (hp wp c)",
                         hp=self.ps, wp=self.ps)

    x, logdet = m_inverse(block, x, kinds, c_projs, s_projs, context)
    logdet = jnp.sum(logdet, axis=0)

    x = einops.rearrange(x, "b (h w) (hp wp c) -> b (h hp) (w wp) c",
                         hp=self.ps, wp=self.ps,
                         h=np.round(x.shape[1] ** 0.5).astype(np.int32))

    return x, logdet


def get_channels_coupling_init(depth, image_shape, ps, proj_kinds):
  """Channel-wise coupling projection init."""
  assert image_shape[-3] % ps == 0, f"Shape and patch size: {image_shape}, {ps}"
  assert image_shape[-2] % ps == 0, f"Shape and patch size: {image_shape}, {ps}"
  c = image_shape[-1] * ps * ps  # number of dims

  def _init(k, dtype):
    w = jnp.zeros((depth, c, c), dtype=dtype)
    for i, kind in enumerate(proj_kinds):
      if kind == "random":
        p = jax.random.permutation(jax.random.fold_in(k, i), c)
        w = w.at[jnp.ones_like(p) * i, p, jnp.arange(c)].set(1.0)
      elif kind == "zero":  # placeholder all-zero proj
        pass
      else:
        raise ValueError(f"Unknown coupling kind: {kind}")
    return w

  return _init


def get_spatial_coupling_init(depth, image_shape, ps, proj_kinds):
  """Spatial coupling projection init."""
  assert image_shape[-3] % ps == 0, f"Shape and patch size: {image_shape}, {ps}"
  assert image_shape[-2] % ps == 0, f"Shape and patch size: {image_shape}, {ps}"
  nh = image_shape[-3] // ps
  nw = image_shape[-2] // ps
  n = nh * nw  # number of tokens in transformer

  def _init(k, dtype):
    del k
    w = jnp.zeros((depth, n, n), dtype=dtype)
    for i, kind in enumerate(proj_kinds):
      if kind.startswith("vstripes"):
        idx1 = jnp.arange(n)[::2]
        idx2 = jnp.arange(1, n)[::2]
      elif kind.startswith("hstripes"):
        idx1 = jnp.where((jnp.arange(n) // nw) % 2 == 0, size=n//2)[0]
        idx2 = jnp.where((jnp.arange(n) // nw) % 2 == 1, size=n//2)[0]
      elif kind.startswith("checkerboard"):
        vals = jnp.arange(n).reshape([nh, nw]) + jnp.arange(nh).reshape([nh, 1])
        idx1 = jnp.where((vals.flatten() % 2) == 0, size=n//2)[0]
        idx2 = jnp.where((vals.flatten() % 2) == 1, size=n//2)[0]
      elif kind == "zero":  # placeholder all-zero proj
        continue
      else:
        raise ValueError(f"Unknown coupling kind: {kind}")

      idx1, idx2 = (idx2, idx1) if kind.endswith("-inv") else (idx1, idx2)
      w = w.at[i, idx1, jnp.arange(n//2)].set(1)
      w = w.at[i, idx2, jnp.arange(n//2, n)].set(1)
    return w

  return _init


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


