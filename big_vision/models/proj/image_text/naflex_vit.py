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

"""NaFlex ViT = NaViT + FlexiViT.

Based on:
* FlexiViT: https://arxiv.org/abs/2212.08013
* NaViT: https://arxiv.org/abs/2307.06304
"""

import re
from big_vision.models import vit
import big_vision.models.proj.image_text.utils as it_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def _decode_posemb(posemb):
  if (m := re.fullmatch(r"learn_2d(\(\d+\))", posemb)):
    grid_size = int(m.groups()[0][1:-1])
    return "learn_2d", grid_size
  return posemb, None


def _pos_emb_resize(pos_emb, shapes, coords, l):
  """Resizes the positional embeddings to match the input image size.
  
  Args:
    pos_emb: Positional embeddings.
    shapes: Image shapes (usually `coords.max(axis=1) + 1`).
    coords: Patch coordinates.
    l: Maximum number of patches per side. Necesary in order to have a static
      return shape.

  Setting l to 64 is a heuristic. Ideally, we would use
  `l = tokens.shape[1]` here, but that requires too much memory,
  especially for high-resolution inputs. Using a lower value
  effectively limits the maximum resolution to `l x patch_size`.
  Resolutions above that will lead to NaNs in the positional
  embeddings and NaN model outputs.
  Note: this value can be adjusted post-hoc without retraining.

  Returns:
    Postional embeddings for every patch.
  """

  def resize_fn(shape, coords):
    emb = jax.image.scale_and_translate(
        pos_emb,
        shape=(l, l, pos_emb.shape[-1]),
        spatial_dims=(0, 1),
        scale=shape / jnp.asarray(pos_emb.shape[:2]),
        translation=jnp.asarray([0, 0]),
        method="bilinear", antialias=True)
    gather_dim = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0, 1),
        start_index_map=(0, 1, 2)
    )
    return jax.lax.gather(
        emb,
        jnp.pad(coords, [[0, 0], [0, 1]]),
        gather_dim,
        [1, 1, emb.shape[-1]],
        mode="fill")
  return it_utils.batch_shmap(
      jax.vmap(resize_fn, in_axes=(0, 0), out_axes=0),
      shapes, coords)


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, mask=None, deterministic=True):
    if mask is not None:
      mask = mask[..., None, :, :]  # Broadcast mask along the head dim.

    out = {}
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    y = nn.LayerNorm()(x)
    y = out["sa"] = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
    )(y, y, mask=mask)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = vit.MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, mask=None, deterministic=True):
    out = {}

    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(3,),  # 0=self, 3=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
          )
      x, scan_out = nn.scan(
          block,
          variable_axes={"params": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth)(
              name="encoderblock",
              dtype_mm=self.dtype_mm,
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout=self.dropout)(x, mask, deterministic)
      for lyr in range(self.depth):
        out[f"block{lyr:02d}"] = jax.tree.map(lambda o, l=lyr: o[l], scan_out)
    else:
      # Input Encoder
      for lyr in range(self.depth):
        block_cur = Encoder1DBlock(
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim, num_heads=self.num_heads,
            dropout=self.dropout)
        x, out[f"block{lyr:02d}"] = block_cur(x, mask, deterministic)
      out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x, mask=None):
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    if mask is not None:
      mask = mask[..., None, None, :]  # Add query and head dims.

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x, mask=mask)

    y = nn.LayerNorm()(x)
    x = x + vit.MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class _Model(nn.Module):
  """ViT model."""

  num_classes: int | None = None
  width: int = 768
  depth: int = 12
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  rep_size: int | bool = False
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True
  scan: bool = False
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  posemb: str = "learn_2d(64)"
  nposemb: int | None = None  # Needs to be overwritten

  patchln_pre: bool = False
  patchln_post: bool = False

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}

    patches, ptype, yabs, xabs = image
    patches = jnp.asarray(patches, self.dtype_mm)  # BN(hw3) of float32

    if self.patchln_pre:
      patches = nn.LayerNorm(name="patchln_pre")(patches)

    # Embed the patches.
    tokens = out["stem"] = nn.Dense(
        self.width, name="embedding", dtype=self.dtype_mm)(patches)

    if self.patchln_post:
      tokens = nn.LayerNorm(name="patchln_post")(tokens)

    x = tokens
    posemb, posemb_grid_size = _decode_posemb(self.posemb)
    if posemb == "learn_2d":
      posembs = self.param(
          "pos_embedding",
          nn.initializers.normal(stddev=1/np.sqrt(self.width)),
          (self.nposemb, self.nposemb, self.width), self.dtype_mm)
      coords = jnp.stack([yabs, xabs], axis=-1)
      shapes = coords.max(axis=1) + 1
      # See comment in `_pos_emb_resize` for details.
      x += _pos_emb_resize(posembs, shapes, coords, posemb_grid_size or 64)
    else:
      raise ValueError(f"Unknown posemb: '{self.posemb}'")

    out["with_posemb"] = x

    # Only use patch tokens in self-attention:
    sa_mask = ptype == 1  # 1 == patch (pad is 0).
    sa_mask = jnp.logical_and(sa_mask[..., :, None], sa_mask[..., None, :])
    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        name="Transformer")(
            x, mask=sa_mask, deterministic=not train)
    out["encoded"] = x

    # Ignore the padding tokens when pooling:
    pool_mask = (ptype == 1)  # 1 == patch (not pad)
    if self.pool_type == "map":
      maphead = MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim)
      x = maphead(x, mask=pool_mask)
    elif self.pool_type == "gap":
      pool_mask = pool_mask[..., None]
      x = jnp.sum(x * pool_mask, axis=1) / jnp.sum(pool_mask, axis=1)
    elif self.pool_type == "max":
      # Tested in (internal link)
      pool_mask = pool_mask[..., None]
      ignore = jnp.where(pool_mask, 0, jnp.finfo(x.dtype).min)
      x = jnp.max(pool_mask * x + ignore, axis=1)
    elif self.pool_type == "none":
      pass
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")
    out["head_input"] = x

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name="pre_logits")
      x = nn.tanh(hid(x))

    out["pre_logits"] = x

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="head", **kw)
      x = out["logits"] = head(x)

    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**vit.decode_variant(variant), **kw})

load = vit.load
