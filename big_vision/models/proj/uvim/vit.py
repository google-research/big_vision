# Copyright 2022 Big Vision Authors.
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

"""VQ-VAE autoencoder with ViT backbone."""

import functools
from typing import Mapping, Optional, Sequence, Union

from big_vision import utils
from big_vision.models import common
from big_vision.models import vit

import einops
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np


partial = functools.partial

# Multiplicative perturbation applied to codewords when doing the split.
# Note, the multiplicative pertubation is not perfectly symmetric and rep.
# applications can shrink the embedding. However, in practice it does not matter
# for the value we use.
PERTURB = 0.001


# The function below takes a vector `x` and a dictioniary of vectors `e` as an
# input. It then returns a "quantized" version of x (namely the closest to `x`
# vector from `e`) and its index in `e` as well.
# On top of this, it has two extra features:
#   1. Double `vmap` vectorizes this function to operate on many `x` vectors.
#      More concretely, we add two extra dimensions (batch and space) to `x`.
#      Also note we compute euclidian distance in a decomposed way, because it
#      makes it more efficient for vmapping.
#   2. `quantize` is a "discrete" operation, so it does not have a gradient for
#      `x`. So we implement a so-called "straight-through" gradient estimator
#      using `stop_gradient` magic. It does not affect forward pass, but changes
#      the gradient.
@partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
@partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
def quantize(x, e):
  dist = jnp.sum(x * x)[None] - 2 * x.dot(e.T) + jnp.sum(e * e, axis=1)
  idx = jnp.argmin(dist)
  x_q = jax.lax.stop_gradient(e[idx] - x) + x  # just `e[idx]` for the fwd pass.
  return x_q, idx


def split_the_most_frequent_embedding(state):
  """Splits most frequent embedding into two and eliminates least frequent.

  Args:
    state: a dict. that contains current jax rng, embeddings and their counts.

  Returns:
    New dict. with the updated jax rng, embeddings and counts.
  """
  rng, e, c = state["rng"], state["dictionary"], state["counts"]
  rng, rng_local = jax.random.split(rng)

  i_max = jnp.argmax(c)
  i_min = jnp.argmin(c)

  e = e.at[i_min].set(
      e[i_max] * jax.random.uniform(rng_local, (e.shape[1],), jnp.float32,
                                    1.0-PERTURB, 1.0+PERTURB))

  c = c.at[i_min].set(c[i_max] / 2.0)
  c = c.at[i_max].set(c[i_max] / 2.0)

  e = e.at[i_min].set(e[i_min] / 2.0)
  e = e.at[i_max].set(e[i_max] / 2.0)

  return {"rng": rng, "dictionary": e, "counts": c}


class Model(nn.Module):
  """ViT model."""

  inputs: Mapping[str, Sequence[int]]
  outputs: Mapping[str, Sequence[int]]
  input_size: Sequence[int] = (256, 256)
  patch_size: Sequence[int] = (8, 8)
  code_len: int = 256
  width: int = 768
  enc_depth: int = 6
  dec_depth: int = 6
  mlp_dim: Optional[int] = None
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  reinit: Optional[Sequence[str]] = None
  head_zeroinit: bool = True
  dict_size: int = 512  # Number of words in dict.
  codeword_dim: Optional[int] = None
  dict_momentum: float = 0.995  # Exp. moving average coeff. for dict. learning.
  quantize: bool = True
  # Useful to set to None when running without pmap, e.g. testing.
  statistics_axis_name: str = "batch"
  # Threshold for the discounted count after which the codeword will be
  # considered unused. For the `dict_momentum` param of 0.995 the codeword
  # should not be present in ~500 batches in a row.
  min_count: float = 0.1  # ~= 0.995 ** 500
  with_encoder_ctx: bool = False
  with_decoder_ctx: bool = False
  code_dropout: str = "none"
  bottleneck_resize: bool = False
  zero_decoder_seq: bool = False

  def setup(self):

    self.grid_size = np.array(self.input_size) // np.array(self.patch_size)

    self.embeddings = {
        k: nn.DenseGeneral(features=(self.width,), axis=range(-len(shape), 0),
                           name=f"embedding_{k}")
        for k, shape in self.inputs.items()
    }

    kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
    self.heads = {
        k: nn.DenseGeneral(features=shape, name=f"head_{k}", **kw)
        for k, shape in self.outputs.items()
    }

    if self.with_encoder_ctx:
      self.stem_conv_ctx_enc = nn.Conv(
          self.width, self.patch_size, strides=self.patch_size,
          padding="VALID", name="ctx_enc_embedding")

    if self.with_decoder_ctx:
      self.stem_conv_ctx_dec = nn.Conv(
          self.width, self.patch_size, strides=self.patch_size,
          padding="VALID", name="ctx_dec_embedding")

    self.pos_embedding_encoder = vit.get_posemb(
        self, self.posemb, self.grid_size, self.width, "pos_embedding_encoder")
    self.encoder = vit.Encoder(
        depth=self.enc_depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        name="encoder")

    if not self.bottleneck_resize:
      self.bottleneck_downsample = self.param(
          "bottleneck_downsample",
          nn.initializers.xavier_uniform(),
          (np.prod(self.grid_size), self.code_len))

    norm_init = nn.initializers.normal(stddev=1.0 / np.sqrt(self.dict_size))
    self.dictionary = self.variable(
        "state", "dictionary",
        lambda shape: norm_init(self.make_rng("state"), shape),
        (self.dict_size, self.codeword_dim or self.width))
    self.counts = self.variable("state", "counts", jnp.ones, (self.dict_size,))

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
        name="decoder")

    self.encoder_head = nn.Dense(self.codeword_dim or self.width)
    self.decoder_stem = nn.Dense(self.width)

  def get_codewords(self):
    e = self.dictionary.value / self.counts.value[:, None]
    e = e / jnp.linalg.norm(e, axis=-1, keepdims=True)
    return e

  def encode(self, x, *, ctx=None, train=False, update_dict=True):
    out = {}

    out["stem"] = {}
    for key, embed in self.embeddings.items():
      out["stem"][key] = embed(x[key])
    x = sum(out["stem"].values())

    if self.with_encoder_ctx:
      ctx_tokens = self.stem_conv_ctx_enc(ctx)
      ctx_tokens = einops.rearrange(ctx_tokens, "b h w c -> b (h w) c")
      x = x + ctx_tokens

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

    x = jax.nn.standardize(x, axis=-1)
    x_pre_q = out["bottleneck"] = x
    e = self.get_codewords()
    x, idx = quantize(x, e)
    out["bottleneck_q"] = x
    out["code"] = idx

    # Implements explicit dictionary learning algo outlined in the VQ-VAE paper.
    # We slightly deviate from the papers formulation, as we find it confusing,
    # especially in the multi-host scenario. What is implemented below can be
    # seen as computing discounted counts and sums of all embeddings.
    if train:
      # Compute counts and sum(x) of code in the global batch.
      counts = jnp.zeros(self.dict_size, dtype=jnp.int32)
      counts = counts.at[idx].add(1)

      # Below we introduce redundant stop_gradient, because jax' dead code
      # elimination for our program's gradient fails to infer that the code
      # below does not require gradient computation.
      # Relevant github issue: https://github.com/google/jax/issues/9042.
      # TODO: remove stop_gradient when the bug is fixed.
      x_sum = jnp.zeros_like(self.dictionary.value)
      x_sum = x_sum.at[idx].add(jax.lax.stop_gradient(x_pre_q))

      if self.statistics_axis_name:
        counts = jax.lax.psum(counts, axis_name=self.statistics_axis_name)
        x_sum = jax.lax.psum(x_sum, axis_name=self.statistics_axis_name)

      out["codebook_max_ratio"] = jnp.max(counts) / jnp.sum(counts)
      out["codebook_zeros_ratio"] = jnp.sum(counts == 0) / len(counts)

      if update_dict:
        self.counts.value = self.counts.value * self.dict_momentum + counts
        self.dictionary.value = (self.dictionary.value * self.dict_momentum +
                                 x_sum)

        state = {"dictionary": self.dictionary.value,
                 "counts": self.counts.value,
                 "rng": self.make_rng("vqvae")}
        new_state = jax.lax.while_loop(
            lambda state: jnp.any(state["counts"] < self.min_count),
            split_the_most_frequent_embedding,
            state)
        self.counts.value = new_state["counts"]
        self.dictionary.value = new_state["dictionary"]

    if not self.quantize:
      x = x_pre_q
      out["bottleneck_q"] = x
    return x, out

  def decode(self, x, ctx=None, discrete_input=False, train=False):
    out = {}

    if discrete_input:
      e = self.get_codewords()
      x = e[x]

    if self.zero_decoder_seq:
      x = jnp.zeros_like(x)

    if train and self.code_dropout != "none":
      importance = jnp.linspace(1.0, 0.0, self.code_len + 2)[1:-1]
      thr = jax.random.uniform(self.make_rng("dropout"), x.shape[:1])
      mask = importance[None, :] > thr[:, None]
      if self.code_dropout == "random":
        mask = jax.random.permutation(
            self.make_rng("dropout"), mask, axis=-1, independent=True)
      x = x * mask[:, :, None]

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

    if self.with_decoder_ctx:
      ctx_tokens = self.stem_conv_ctx_dec(ctx)
      ctx_tokens = einops.rearrange(ctx_tokens, "b h w c -> b (h w) c")
      x = x + ctx_tokens

    x, _ = self.decoder(x + self.pos_embedding_decoder)

    out["logits"] = {}
    for key, head in self.heads.items():
      out["logits"][key] = head(x)

    return out["logits"], out

  def __call__(self, x, *, ctx=None, train=False, update_dict=True):
    x, out_enc = self.encode(x, ctx=ctx, train=train, update_dict=update_dict)
    x, out_dec = self.decode(x, ctx=ctx, train=train)
    return x, {**out_enc, **out_dec}


def load(init_params, init_file, model_params=None, dont_load=()):
  """Loads params from init checkpoint and merges into init_params."""
  del model_params
  ckpt = flax.core.unfreeze(utils.load_checkpoint(None, init_file))
  params = {"params": ckpt["params"], "state": ckpt["state"]}
  params = flax.training.checkpoints.convert_pre_linen(params)
  # Fix old-style param name.
  if "Encoder" in params["params"]:
    p = params["params"]
    p["encoder"] = p.pop("Encoder")
    p["decoder"] = p.pop("Decoder")
    params["params"] = p
  if init_params is not None:
    params = common.merge_params(params, init_params, dont_load)
  return params["params"], params["state"]
