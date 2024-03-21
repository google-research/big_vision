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

"""Decoder-only and encoder-decoder GIVT model.

Used abbreviations for dimension annotations:
  B: batch size.
  E: embedding size.
  L: (soft) token sequence length.
  D: soft token dimension.
  P: number of patches (extracted by a ViT encoder in GIVT-based UViM)
"""

import enum
import itertools
from typing import Literal, Optional, Sequence, Any, Mapping

from absl import logging
from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
import distrax
import einops
import flax.linen as nn
from flax.linen import partitioning
import jax
import jax.numpy as jnp
import numpy as np


class _SpecialLabel(enum.Enum):

  MASK = "mask"
  NOMASK = "nomask"
  REPLACE = "replace"
  NOLABEL = "nolabel"  # For CFG


def _random_mask_with_ratios(rng, ratios: jax.Array, seq_len: int):
  """Generates masks where a fraction of tokens is uncovered.

  Args:
    rng: RNG.
    ratios: Ratios, must be a 1D matrix of shape (B,). Values must be in
      [0, 1], and indicate at ratios[i] how many of the i-th tokens are
      uncovered (ie. equal to `True`).
    seq_len: How many tokens this mask has to cover.

  Returns:
    Mask of dtype bool, shape (B, L).

  Raises:
    ValueError: Incorrect inputs.
  """
  if ratios.ndim != 1:
    raise ValueError("Ratios must have shape (B,)!")
  ratios = jnp.clip(ratios, 0, 1)
  indices = jnp.arange(seq_len, dtype=jnp.float32)  # Shape: (L,)
  ratios = ratios[:, jnp.newaxis] * seq_len  # Shape: (B, 1)
  # This is a binary array where the first ratios * seq_len positions are True
  mask = (indices < ratios).astype(jnp.bool_)  # Shape: (B, L)
  # Shuffle to a actual mask.
  return jax.random.shuffle(rng, mask, axis=-1)


def apply_mask_schedule(ratio: float | jax.Array, method: str) -> jax.Array:
  """Generate a mask rate by scheduling mask functions R."""
  if method == "cosine":
    mask_ratio = jax.lax.cos(jnp.pi / 2. * ratio)
  elif "pow:" in method:
    exponent = float(method.replace("pow:", ""))
    mask_ratio = 1. - ratio**exponent
  else:
    raise NotImplementedError(method)
  # Clamps mask into [epsilon, 1)
  mask_ratio = jnp.clip(mask_ratio, 1e-6, 1.)
  return mask_ratio


class EncoderDecoderBlock(nn.Module):
  """Transformer encoder-decoder layer."""
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.
  decode: bool = False

  @nn.compact
  def __call__(
      self,
      targets: jax.Array,
      encoded: jax.Array | None = None,
      decoder_mask: jax.Array | None = None,
      deterministic: bool = True,
  ) -> tuple[jax.Array, jax.Array]:
    """Applies EncoderDecoderBlock module.

    Args:
      targets: target text embeddings [B, L, D].
      encoded: encoded image patches from encoder [B, P, E].
      decoder_mask: decoder self-attention mask.
      deterministic: bool, deterministic or not (to apply dropout).

    Returns:
      output after transformer encoder-decoder block [B, L, E].
    """
    # Helper function for axis annotation.
    def wlc(f):
      dim_names = ("act_batch", "act_len", "act_emb")
      return nn.with_logical_constraint(f, dim_names)
    # Decoder block.
    x = wlc(nn.LayerNorm(name="LayerNorm1", use_bias=False)(targets))
    x = wlc(nn.SelfAttention(
        num_heads=self.num_heads, use_bias=False, broadcast_dropout=False,
        dropout_rate=self.dropout_rate, decode=self.decode, name="SelfAttn")(
            x, decoder_mask, deterministic=deterministic))
    x = wlc(nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic))
    x = wlc(x + targets)

    if encoded is None:
      y = x
    else:
      # Encoder-Decoder block.
      y = wlc(nn.LayerNorm(name="LayerNorm2", use_bias=False)(x))
      y = wlc(nn.MultiHeadDotProductAttention(
          num_heads=self.num_heads, use_bias=False, broadcast_dropout=False,
          dropout_rate=self.dropout_rate, name="CrossAttn")(
              y, encoded, deterministic=deterministic))
      y = wlc(
          nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic))
      y = wlc(y + x)

    # MLP block.
    z = wlc(nn.LayerNorm(name="LayerNorm3", use_bias=False)(y))
    z = wlc(vit.MlpBlock(mlp_dim=self.mlp_dim, dropout=self.dropout_rate,
                         name="MLP")(z, deterministic=deterministic))

    # nn.scan requires a carry (second element in tuple)
    out = wlc(y + z)
    return out, out


class Decoder(nn.Module):
  """Transformer decoder model with optional cross-attention."""
  emb_dim: int
  mlp_dim: int
  num_heads: int
  num_layers: int
  out_dim: int
  seq_len: int
  style: Literal["ar", "masked"]
  dropout_rate: float = 0.
  zero_embedding_init: bool = False

  scan: bool = False
  remat_policy: str = "nothing_saveable"

  @nn.compact
  def __call__(
      self,
      targets: jax.Array,
      encoded: jax.Array | None = None,
      decoder_mask: jax.Array | None = None,
      decode: bool = False,
      deterministic: bool = True,
      return_reps: bool = False,
  ) -> jax.Array | tuple[jax.Array, Mapping[str, jax.Array]]:
    """Applies Transformer model on the inputs.

    Args:
      targets: target text tokens [B, L].
      encoded: encoded sequence from an encoder [B, P, E].
      decoder_mask: decoder self-attention mask.
      decode: bool, whether to perform fast autoregressive decoding with cache.
      deterministic: bool, deterministic or not (to apply dropout).
      return_reps: bool, whether to return intermediate representations.

    Returns:
      output of a transformer decoder [B, L, out_dim], where out_dim is usually
      a multiple of D.
    """
    if self.style == "masked" and decode:
      raise ValueError("Cannot run masked model in cached mode!")

    pos_emb = vit.get_posemb(
        self, "learn", self.seq_len, self.emb_dim,
        "pos_emb")

    y = common.AddPositionEmbs(
        decode=decode, name="PosEmbedTargets")(targets, pos_emb)

    out = {}
    if self.scan:
      # Mostly followed
      # https://github.com/google/maxtext/blob/4d99e30b3e0e0cb1d1aa11c7db7fffe18e301498/MaxText/layers.py#L1126
      # for the scanned version.

      # 1. remat
      enc_dec_block_remat = nn.remat(
          EncoderDecoderBlock,
          prevent_cse=False,
          static_argnums=(-1, -2),
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None))
      # 2. scan
      initializing = self.is_mutable_collection("params")
      param_scan_axis = 1
      params_spec = (param_scan_axis if initializing
                     else partitioning.ScanIn(param_scan_axis))
      dec_scanned = nn.scan(enc_dec_block_remat,
                            variable_axes={
                                "params": params_spec,
                                "cache": 0,
                            },
                            split_rngs={"params": True, "dropout": True},
                            in_axes=nn.broadcast,
                            length=self.num_layers)
      # 3. fprop
      y, out = dec_scanned(num_heads=self.num_heads, mlp_dim=self.mlp_dim,
                           dropout_rate=self.dropout_rate, decode=decode,
                           name="EncDecBlock")(
                               y, encoded, decoder_mask, deterministic)
      # Extracting the intermediate representation from the stacked activation
      # tensor `out`, which is a [num_layers, B, L, E] tensor. Indexing along
      # the first axis to extract individual layers, and then averaging across
      # the second axis, which corresponds to the sequence dimension after
      # indexing.
      assert out.shape[0] == self.num_layers and (
          decode or out.shape[2] == self.seq_len), (
              (out.shape, self.num_layers, self.seq_len))
      out = {f"block{l}_rep": jnp.mean(out[l], axis=1)
             for l in range(self.num_layers)}
    else:
      for lyr in range(self.num_layers):
        y, _ = EncoderDecoderBlock(
            num_heads=self.num_heads, mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate, decode=decode,
            name=f"EncDecBlock{lyr}")(y, encoded, decoder_mask=decoder_mask,
                                      deterministic=deterministic)
        out[f"block{lyr}_rep"] = jnp.mean(y, axis=1)
    y = nn.LayerNorm(name="LayerNorm")(y)
    out["pre_logits"] = jnp.mean(y, axis=1)

    logits = nn.Dense(
        self.out_dim,
        kernel_init=nn.initializers.zeros,
        name="LogitsDense",
    )(y)
    out["logits"] = logits
    if return_reps:
      return logits, out
    return logits


class Model(nn.Module):
  """GIVT model supporting decoder-only and encoder-decoder applications."""
  num_heads: int = 8
  # num_layers = 0 means no encoder
  num_layers: int = 0
  num_decoder_layers: int = 6
  mlp_dim: int = 2048
  enc_dropout_rate: float = 0.
  dec_dropout_rate: float = 0.
  # Decoder params:
  emb_dim: int = 512
  num_labels: Optional[int] = 1000
  seq_len: int = 256
  # Encoder params:
  patches: Sequence[int] = (16, 16)
  input_size: Sequence[int] = (256, 256)
  posemb_type: Literal["learn", "sincos2d"] = "learn"
  zero_decoder_seq: bool = False
  style: Literal["ar", "masked"] = "ar"

  zero_embedding_init: bool = False

  num_mixtures: int = 4
  multivariate: bool = False
  out_dim: int = 32
  scale_tol: float = 1e-6

  # Mask specific params.
  mask_schedule_train: str = "cosine"
  # Results in at least 40% masked tokens with cosine.
  min_masking_rate_training: float = 0.3

  # How to fuse mask at input:
  # - replace: replace token[masked] with lookup(MASK)
  # - concat: replace token[mask] with lookup(REPLACE) and concat either
  #   lookup(NOMASK) or lookup(MASK).
  mask_style: str = "replace"

  # Set to >0 for CFG support.
  drop_labels_probability: float = 0.0

  fix_square_plus: bool = False

  # If True, and mixture >1, create a GMM per channel. Otherwise, create
  # a GMM of `dim`-dimensional Gaussians.
  per_channel_mixtures: bool = True

  scan: bool = False
  remat_policy: str = "nothing_saveable"

  @property
  def has_encoder(self) -> bool:
    return self.num_layers > 0

  @property
  def num_logits(self) -> int:
    if self.multivariate:
      assert self.num_mixtures == 1
      # d**2 covariance, d means.
      # Note: `round` makes pytype happy.
      return round(self.out_dim ** 2) + self.out_dim

    elif self.per_channel_mixtures:
      # One (mu, sigma, pi) per output dimension and mixture component.
      # Note that we predict a distribution for each output dimensions in
      # parallel.
      return 3 * self.num_mixtures * self.out_dim

    else:
      # Mixture weights plus mean/scale per mixture
      return self.num_mixtures + 2 * self.num_mixtures * self.out_dim

  def setup(self) -> None:
    assert self.posemb_type == "learn"
    assert self.num_mixtures > 0

    if self.multivariate and self.num_mixtures != 1:
      raise ValueError("Cannot do multivariate GMM!")

    if self.num_layers > 0:
      grid_size = np.array(self.input_size) // np.array(self.patches)

      self.pos_emb_for_encoder = vit.get_posemb(
          self, self.posemb_type, grid_size, self.emb_dim,
          "pos_embedding_encoder")

      self.conv = nn.Conv(self.emb_dim, self.patches, padding="VALID",
                          strides=self.patches, name="EmbedPatches")

      self.encoder = vit.Encoder(
          depth=self.num_layers,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.enc_dropout_rate,
          scan=self.scan,
          remat_policy=self.remat_policy,)
    else:
      self.encoder = None

    # Iterator that will lead free label IDs.
    next_label = itertools.count(self.num_labels or 0)
    special_labels = {}

    if self.style == "ar":
      pass
    elif self.style == "masked":
      if self.mask_style == "replace":
        special_labels = {_SpecialLabel.MASK: next(next_label)}
      elif self.mask_style == "concat":
        special_labels = {
            _SpecialLabel.MASK: next(next_label),
            _SpecialLabel.NOMASK: next(next_label),
            _SpecialLabel.REPLACE: next(next_label),
        }
      else:
        raise NotImplementedError(self.mask_style)
    else:
      raise NotImplementedError(self.style)

    if self.drop_labels_probability > 0:
      special_labels[_SpecialLabel.NOLABEL] = next(next_label)

    self.special_labels = special_labels
    lookup_size = (self.num_labels or 1) + len(self.special_labels)

    self.labels_emb = nn.Embed(
        lookup_size,
        self.emb_dim,
        name="EmbedLabels",
        embedding_init=nn.initializers.zeros
        if self.zero_embedding_init
        else nn.initializers.normal(stddev=1.0),
    )

    self.targets_emb = nn.Dense(self.emb_dim, name="EmbedTargets")

    self.decoder = Decoder(
        num_layers=self.num_decoder_layers or self.num_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        out_dim=self.num_logits,
        # In masked mode, we run with 1 more token at the input.
        seq_len=self.seq_len + int(self.style == "masked"),
        dropout_rate=self.dec_dropout_rate,
        emb_dim=self.emb_dim,
        zero_embedding_init=self.zero_embedding_init,
        style=self.style,
        scan=self.scan,
        remat_policy=self.remat_policy,
    )

  def encode(self, image: jax.Array, train: bool = False) -> jax.Array:
    """Encodes input image or embeddings."""
    emb = self.conv(image)
    patch_embeddings = einops.rearrange(emb, "B PH PW E -> B (PH PW) E")
    encoded, _ = self.encoder(
        patch_embeddings + self.pos_emb_for_encoder, deterministic=not train)
    return encoded

  def embed_labels(
      self,
      labels: jax.Array | None = None,
      batch_size: int | None = None,
  ) -> jax.Array:
    if labels is not None:
      # Embed class label, add a sequence dim (output shape (B, 1, E))
      return self.labels_emb(labels)[:, None, :]

    assert ((self.num_labels == 1 or self.num_labels is None)
            and batch_size is not None)
    # Create [BOS] token embedding
    return self.labels_emb(jnp.zeros((batch_size,), jnp.int32))[:, None, :]

  def prefill(
      self, labels=None, batch_size=None, encoded=None, drop_labels=None
  ):
    labels = self._drop_labels(drop_labels, labels)
    labels_for_prefill = self.embed_labels(labels=labels, batch_size=batch_size)
    return self.decoder(
        labels_for_prefill,
        encoded=encoded,
        decode=True)

  def _decode_ar(
      self,
      targets: jax.Array,
      labels: jax.Array | None = None,
      encoded: jax.Array | None = None,
      decode: bool = False,
      train: bool = False,
  ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
    """Autoregressive decoding."""
    targets_embedded = self.targets_emb(targets)

    if decode:
      decoder_mask = None
    else:
      decoder_mask = nn.make_causal_mask(targets[:, :, 0])
      b = targets.shape[0]
      labels_embedded = self.embed_labels(labels, b)
      assert labels_embedded.shape == (b, 1, self.emb_dim), (
          labels_embedded.shape, (b, 1, self.emb_dim))
      targets_embedded = jnp.concatenate(
          [labels_embedded, targets_embedded[:, : -1]], axis=1)

    logits, out = self.decoder(
        targets_embedded,
        encoded=encoded,
        decoder_mask=decoder_mask,
        decode=decode,
        deterministic=not train,
        return_reps=True)

    return logits, out

  def _get_special_label(self, size, label: _SpecialLabel):
    return self.labels_emb(
        jnp.full(size, self.special_labels[label], jnp.int32)
    )

  def _decode_masked(
      self,
      targets,
      input_mask,
      labels=None,
      encoded=None,
      train=False,
  ):
    """Masked decoding."""
    b, s, _ = targets.shape
    assert input_mask.shape == (b, s)

    if self.mask_style == "replace":
      targets_embedded = jnp.where(
          input_mask[:, :, None],
          self._get_special_label((b, s), _SpecialLabel.MASK),
          self.targets_emb(targets),
      )
    elif self.mask_style == "concat":
      masks = jnp.where(
          input_mask[:, :, None],
          self._get_special_label((b, s), _SpecialLabel.MASK),
          self._get_special_label((b, s), _SpecialLabel.NOMASK),
      )
      embedded_targets = self.targets_emb(targets)
      targets_embedded = jnp.where(
          input_mask[:, :, None],
          self._get_special_label((b, s), _SpecialLabel.REPLACE),
          embedded_targets,
      )
      # Only take half of each to get the right embedding size.
      targets_embedded = jnp.concatenate(
          [masks[..., ::2], targets_embedded[..., ::2]], axis=-1
      )
    else:
      raise ValueError(self.mask_style)

    labels_embedded = self.embed_labels(labels, b)
    assert labels_embedded.shape == (b, 1, self.emb_dim)
    # Note that we do not truncate the input here, so this has shape
    # (B, L+1, E).
    targets_embedded = jnp.concatenate(
        [labels_embedded, targets_embedded], axis=1)

    logits = self.decoder(
        targets_embedded,
        encoded=encoded,
        decoder_mask=None,
        decode=False,
        deterministic=not train)

    logits = logits[:, 1:, ...]  # Remove class label
    assert logits.shape[:2] == (b, s)
    return logits

  def _drop_labels(self, drop_labels_mask, labels):
    if labels is None:
      return None
    if self.drop_labels_probability >= 0.999:
      logging.warning("Dropping all labels...")
      return jnp.full_like(labels, self.special_labels[_SpecialLabel.NOLABEL])
    if drop_labels_mask is None:
      return labels
    assert _SpecialLabel.NOLABEL in self.special_labels
    nolabel = jnp.full_like(
        labels, self.special_labels[_SpecialLabel.NOLABEL]
    )
    return jnp.where(drop_labels_mask, nolabel, labels)

  def decode(
      self,
      targets: jax.Array,
      labels: jax.Array | None = None,
      encoded: jax.Array | None = None,
      decode: bool = False,
      train: bool = False,
      max_decode_length: int | None = None,
      input_mask: jax.Array | None = None,
      drop_labels: jax.Array | None = None,
      return_reps: bool = False,
  ) -> jax.Array | tuple[jax.Array, Mapping[str, jax.Array]]:
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      targets: target text tokens [B, L, out_dim].
      labels: optional class labes, [B].
      encoded: encoded image patches from encoder [B, P, E].
      decode: whether to prepare and use an autoregressive cache.
      train: whether it is training.
      max_decode_length: optional max length for positional embeddings.
      input_mask: If given, mask input. Required for style=="masked".
        Shape [B, L], bool tensor. True means the token will be removed
        from the input.
      drop_labels: Drop labels at corresponding locations [B].
      return_reps: whether to return intermediate representations.

    Returns:
      logits array from transformer decoder [B, L, 3 * num_mixtures * out_dim].
    """
    del max_decode_length
    labels = self._drop_labels(drop_labels, labels)
    if self.style == "ar":
      logits, out = self._decode_ar(
          targets, labels, encoded, decode, train)
      if return_reps:
        return logits, out
      return logits
    elif self.style == "masked":
      assert not decode  # Cache not supported.
      assert input_mask is not None
      assert not return_reps  # Not implemented.
      return self._decode_masked(targets, input_mask, labels, encoded, train)
    else:
      raise NotImplementedError(self.style)

  def _square_plus(self, x):
    # Via https://twitter.com/jon_barron/status/1387167648669048833
    if self.fix_square_plus:
      return (x + jnp.sqrt(jnp.square(x) + 4)) / 2
    else:
      return x + jnp.sqrt(jnp.square(x) + 4) / 2

  def get_pdf(
      self,
      logits: jax.Array,
      temperature_scales: float | None = None,
      temperature_probs: float | None = None,
  ) -> distrax.Distribution:
    assert logits.shape[-1] == self.num_logits
    if self.multivariate:
      scales = logits[..., :self.out_dim ** 2]
      locs = logits[..., self.out_dim ** 2:]
      assert locs.shape[-1] == self.out_dim
      scales = self._square_plus(scales)
      # Turn into a square matrix.
      *leading, _ = scales.shape
      scales = scales.reshape(*leading, self.out_dim, self.out_dim)
      # Make sure the diagonals are non zero.
      diag_scale_tol = jnp.eye(self.out_dim) * self.scale_tol
      scales = jnp.maximum(scales, diag_scale_tol)
      if (t := temperature_scales) is not None:
        scales = scales * t

      # Note that there is `tfd.MultivariateNormalFullCovariance`` but it just
      # calls linalg.cholesky on the covariance and then uses the
      # MultivariateNormalTri class.  Using ... direcly avoids having to
      # construct a hermetian matrix.
      #
      # Note that only the lower triag part of `scales` is used by applying
      # jnp.tril. The other elements are replaced with zeros.
      #
      # Note on output shapes:
      # - .sample() -> shape (..., seq_len, out_dim)
      # - .prob()   -> shape (..., seq_len).
      return distrax.MultivariateNormalTri(locs, scales)

    elif self.per_channel_mixtures:
      # [..., 3 * num_mixtures * out_dim] -> [..., 3 * out_dim, num_mixtures]
      logits = jnp.reshape(logits, logits.shape[: -1] + (-1, self.num_mixtures))
      # 3 tensors with shape [..., out_dim, num_mixtures]
      probs, locs, scales = jnp.split(logits, 3, axis=-2)
      if (t := temperature_probs) is not None:
        probs = probs * t

      # normalize mixture probabilities
      probs = nn.softmax(probs)
      scales = self._square_plus(scales)
      # threshold scale
      scales = jnp.maximum(scales, self.scale_tol)
      if (t := temperature_scales) is not None:
        scales = scales * t

      # Note on output shapes:
      # - .sample() -> shape (..., seq_len, out_dim)
      # - .prob()   -> shape (..., seq_len, out_dim).
      return distrax.MixtureSameFamily(
          mixture_distribution=distrax.Categorical(probs=probs),
          components_distribution=distrax.Normal(loc=locs, scale=scales),
      )
    else:
      *shape, num_logits = logits.shape
      assert num_logits == self.num_logits, (num_logits, self.num_logits)
      prob_logits, other_logits = (
          logits[..., : self.num_mixtures],
          logits[..., self.num_mixtures :],
      )
      if (t := temperature_probs) is not None:
        prob_logits = prob_logits * t
      other_logits = jnp.reshape(
          other_logits, (*shape, self.num_mixtures, 2, self.out_dim)
      )
      locs = other_logits[..., 0, :]
      scales = self._square_plus(other_logits[..., 1, :])

      scales = jnp.maximum(scales, self.scale_tol)  # Threshold scale
      if (t := temperature_scales) is not None:
        scales = scales * t

      # prob_logits has shape (b, seq_len, m)
      # locs/scales has shape (b, seq_len, m, d)
      assert prob_logits.ndim == locs.ndim - 1, (prob_logits.shape, locs.shape)
      assert locs.shape == scales.shape, (locs.shape, scales.shape)

      # Note on output shapes:
      # - .sample() -> shape (..., seq_len, out_dim)
      # - .prob()   -> shape (..., seq_len,)
      # - .nll()   -> shape (..., seq_len,)
      return distrax.MixtureSameFamily(
          mixture_distribution=distrax.Categorical(logits=prob_logits),
          components_distribution=distrax.MultivariateNormalDiag(
              loc=locs, scale_diag=scales
          ),
      )

  def __call__(
      self,
      sequence: jax.Array,
      labels: jax.Array | None = None,
      *,
      image: jax.Array | None = None,
      decode: bool = False,
      input_mask: jax.Array | None = None,
      drop_labels: jax.Array | None = None,
      train: bool = False,
  ) -> tuple[jax.Array, distrax.Distribution]:
    """Applies Transformer model on the inputs.

    Args:
      sequence: batch of sequences [B, L].
      labels: class labels for class conditional generation [B].
      image: batch of images [B, H, W, 3].
      decode: whether to prepare and use an autoregressive cache.
      input_mask: If given, mask input. Required for style=="masked" [B, L].
      drop_labels: If given, drop labels of the corresponding batches [B].
      train: whether it is training.

    Returns:
      logits array from full transformer [B, L, out_dim].
    """
    if self.style == "masked" and input_mask is None:
      raise ValueError("Cannot run masked model without input mask!")

    if self.encoder is not None:
      assert image is not None
      encoded = self.encode(image, train=train)
    else:
      assert image is None
      encoded = None

    logits = self.decode(sequence, labels=labels, encoded=encoded,
                         decode=decode, input_mask=input_mask, train=train)
    pdf = self.get_pdf(logits)
    return logits, pdf

  def get_input_mask_training(
      self,
      rng: jax.Array,
      shape: tuple[int, int],
  ) -> jax.Array | None:
    """Creates a random maask of shape (B, L) for training masked models."""
    if self.style == "ar":
      return None
    b, s = shape
    # Sample b values in [0, 1-min_mask_ratio].
    keep = jax.random.uniform(
        rng, shape=(b,), maxval=1.0 - self.min_masking_rate_training
    )
    mask_ratio = apply_mask_schedule(keep, self.mask_schedule_train)
    return _random_mask_with_ratios(rng, ratios=mask_ratio, seq_len=s)

  def get_input_mask_teacher_forced(
      self,
      shape: tuple[int, int],
  ) -> jax.Array | None:
    """Creates a random maask of shape (B, L) for training masked models."""
    if self.style == "ar":
      return None
    return jnp.zeros(shape, dtype=jnp.bool_)

  def get_drop_labels(
      self,
      rng: jax.Array,
      batch_size: int,
  ) -> jax.Array | None:
    if (p := self.drop_labels_probability) > 0:
      return jax.random.uniform(rng, shape=(batch_size,)) <= p
    else:
      return None


def load(
    init_params: Any,
    init_files: str | Mapping[str, str],
    model_params: Any = None,
    dont_load: Sequence[str] = (),
    resample_encoder_posemb: bool = False,
    trim_decoder_posemb: bool = False,
) -> Any:
  """Loads params from init checkpoint and merges into init_params."""
  del model_params
  if isinstance(init_files, str):
    ckpt_params = utils.load_params(init_files)
    ckpt_params = common.merge_params(ckpt_params, init_params, dont_load)

    if resample_encoder_posemb:
      if init_params and "pos_embedding_encoder" in init_params:
        ckpt_params["pos_embedding_encoder"] = vit.resample_posemb(
            old=ckpt_params["pos_embedding_encoder"],
            new=init_params["pos_embedding_encoder"])

    if trim_decoder_posemb:
      if init_params and "pos_embedding_decoder" in init_params:
        ckpt_params["pos_embedding_decoder"] = (
            ckpt_params["pos_embedding_decoder"][
                :, :init_params["pos_embedding_decoder"].shape[1], :])

  else:
    init_files = {**init_files}  # Shallow copy because we'll pop stuff off.

    enc_init = init_files.pop("encoder", None)
    if enc_init:
      ckpt_params = init_params.copy()
      vit_params = {
          "pos_embedding": ckpt_params["pos_embedding_encoder"],
          "Transformer": ckpt_params["encoder"],
          "embedding": ckpt_params["EmbedPatches"],
      }
      encoder_params = vit.load(
          vit_params, enc_init, model_cfg={},
          dont_load=dont_load)
      ckpt_params["encoder"] = encoder_params["Transformer"]
      ckpt_params["pos_embedding_encoder"] = encoder_params["pos_embedding"]
      ckpt_params["EmbedPatches"] = encoder_params["embedding"]
    else:
      raise ValueError("Only encoder init is supported: {}.".format(init_files))

  return ckpt_params
