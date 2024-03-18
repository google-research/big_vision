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

"""A version of ViT with flexible seqlen ((internal link))."""

from typing import Optional, Sequence

from absl import logging
from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def resample_patchemb(old, new_hw):
  """Resample the weights of the patch embedding kernel to target resolution.

  We resample the patch embedding kernel by approximately inverting the effect
  of patch resizing. Colab with detailed explanation:
  (internal link)
  With this resizing, we can for example load a B/8 filter into a B/16 model
  and, on 2x larger input image, the result will match.
  See (internal link)
  Args:
    old: original parameter to be resized.
    new_hw: target shape (height, width)-only.
  Returns:
    Resized patch embedding kernel.
  """
  assert len(old.shape) == 4, "Four dimensions expected"
  assert len(new_hw) == 2, "New shape should only be hw"
  if tuple(old.shape[:2]) == tuple(new_hw):
    return old

  logging.info("FlexiViT: resize embedding %s to %s", old.shape, new_hw)

  def resize(x_np, new_shape):
    x_tf = tf.constant(x_np)[None, ..., None]
    # NOTE: we are using tf.image.resize here to match the resize operations in
    # the data preprocessing pipeline.
    x_upsampled = tf.image.resize(
        x_tf, new_shape, method="bilinear")[0, ..., 0].numpy()
    return x_upsampled

  def get_resize_mat(old_shape, new_shape):
    mat = []
    for i in range(np.prod(old_shape)):
      basis_vec = np.zeros(old_shape)
      basis_vec[np.unravel_index(i, old_shape)] = 1.
      mat.append(resize(basis_vec, new_shape).reshape(-1))
    return np.stack(mat).T

  resize_mat = get_resize_mat(old.shape[:2], new_hw)
  resize_mat_pinv = np.linalg.pinv(resize_mat.T)

  def resample_kernel(kernel):
    resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
    return resampled_kernel.reshape(new_hw)
  v_resample_kernel = jax.vmap(jax.vmap(resample_kernel, 2, 2), 3, 3)
  return v_resample_kernel(old)


class Patchify(nn.Module):
  """As a class just to match param names with original ViT."""

  patch_size: Sequence[int] = (32, 32)
  width: int = 768
  seqhw: Optional[int] = None

  @nn.compact
  def __call__(self, image, seqhw=None):
    n, h, w, c = image.shape  # pylint: disable=unused-variable

    w_emb = self.param(
        "kernel", nn.initializers.normal(stddev=1/np.sqrt(self.width)),
        (*self.patch_size, c, self.width), image.dtype)
    b_emb = self.param("bias", nn.initializers.zeros, self.width, image.dtype)

    # Compute required patch-size to reach `seqhw` given `image` size.
    seqhw = seqhw or self.seqhw
    if seqhw is None and self.is_initializing():
      patch_size = self.patch_size
    else:
      patch_size = tuple(np.array((h, w)) // np.array((seqhw, seqhw)))

    if patch_size != self.patch_size:
      w_emb = resample_patchemb(old=w_emb, new_hw=patch_size)

    x = jax.lax.conv_general_dilated(
        image, w_emb, window_strides=patch_size, padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return x + b_emb


class _Model(nn.Module):
  """ViT model."""

  num_classes: int
  patch_size: Sequence[int] = (32, 32)
  posemb_size: Sequence[int] = (7, 7)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True

  seqhw: Optional[int] = None

  @nn.compact
  def __call__(self, image, *, seqhw=None, train=False):
    out = {}

    x = out["stem"] = Patchify(
        self.patch_size, self.width, self.seqhw, name="embedding")(image, seqhw)

    # == Flattening + posemb
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    pos_emb = vit.get_posemb(
        self, self.posemb, self.posemb_size, c, "pos_embedding", x.dtype)
    if pos_emb.shape[1] != h * w:
      pos_emb = jnp.reshape(pos_emb, (1, *self.posemb_size, c))
      pos_emb = jax.image.resize(pos_emb, (1, h, w, c), "linear")
      pos_emb = jnp.reshape(pos_emb, (1, h * w, c))

    x = out["with_posemb"] = x + pos_emb

    # == Optional [cls] token
    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    # == Encoder
    n, l, c = x.shape  # pylint: disable=unused-variable

    x, out["encoder"] = vit.Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        name="Transformer")(x)
    encoded = out["encoded"] = x

    if self.pool_type == "map":
      x = out["head_input"] = vit.MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="head", **kw)
      x_2d = out["logits_2d"] = head(x_2d)
      x = out["logits"] = head(x)

    return x, out


def Model(num_classes, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**vit.decode_variant(variant), **kw})


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  init_file = {**vit.VANITY_NAMES, **VANITY_NAMES}.get(init_file, init_file)
  restored_params = utils.load_params(init_file)

  restored_params = vit.fix_old_checkpoints(restored_params)

  # Potentially resize the position embedings if seqlen differs.
  restored_params["pos_embedding"] = vit.resample_posemb(
      old=restored_params["pos_embedding"],
      new=init_params["pos_embedding"])

  # Potentially resize the patch embedding kernel.
  old_patchemb = restored_params["embedding"]["kernel"]
  restored_params["embedding"]["kernel"] = resample_patchemb(
      old=old_patchemb, new_hw=model_cfg.patch_size)

  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load)

  return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    "FlexiViT-L i1k": "gs://big_vision/flexivit/flexivit_l_i1k.npz",
    "FlexiViT-B i1k": "gs://big_vision/flexivit/flexivit_b_i1k.npz",
    "FlexiViT-S i1k": "gs://big_vision/flexivit/flexivit_s_i1k.npz",
    "FlexiViT-B i21k 90ep": "gs://big_vision/flexivit/flexivit_b_i21k_90ep.npz",
    "FlexiViT-B i21k 300ep": "gs://big_vision/flexivit/flexivit_b_i21k_300ep.npz",
    "FlexiViT-B i21k 1000ep": "gs://big_vision/flexivit/flexivit_b_i21k_1000ep.npz",
    "ViT-B/16 i21k": "gs://big_vision/flexivit/vit_b16_i21k_300ep.npz",
    "ViT-B/30 i21k": "gs://big_vision/flexivit/vit_b30_i21k_300ep.npz",
    # pylint: enable=line-too-long
}
