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

"""JetFormer transformer backbone model.

Used abbreviations for dimension annotations:
  B: batch size.
  E: embedding size.
  L: (soft) token sequence length.
  D: soft token dimension.
  P: number of patches (extracted by a ViT encoder in GIVT-based UViM)
"""

from typing import Optional, Sequence, Any, Mapping

from big_vision import utils
from big_vision.models import common
from big_vision.models.ppp import gemma
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


@jax.vmap
def right_align(x, attn_mask, input_mask):
  """Converts input with masked tokens to be right-aligned."""
  # Note: this right align supports padded tokens anywhere. In particular it
  # supports prefilling with variable text prompt padded to tex_len and also
  # prefill the first image token (e.g. BOI).

  # Due to vmap, this is operating in a single example (not batch level).
  assert x.ndim == 2 and attn_mask.ndim == 2 and input_mask.ndim == 1
  assert x.shape[0] == input_mask.shape[0] == attn_mask.shape[0]
  assert attn_mask.shape[0] == attn_mask.shape[1]
  if x.shape[0] == 0:
    return x, attn_mask, input_mask

  # Compute the final position of each token in the sequence.
  # Tokens with mask==False get assigned position "-1" which ends up being
  # an empty one_hot vector and therefore not used in the permuted version.
  m_cumsum = jnp.cumsum(input_mask)
  seqlen = m_cumsum[-1]
  x_pos = (x.shape[0] - seqlen + m_cumsum) * input_mask - 1

  # Permute x and input mask using desired x_pos of each token.
  x_permut = jax.nn.one_hot(x_pos, x.shape[0], dtype=jnp.bool)
  x = jnp.einsum("n...,nm->m...", x, x_permut).astype(x.dtype)

  # Attention mask has shape [A, B] indicating token A can attent to token B.
  # We need to permute both A and B to reflect the new token positions.
  attn_mask = jnp.einsum("nb,nm->mb", attn_mask, x_permut)
  attn_mask = jnp.einsum("bn,nm->bm", attn_mask, x_permut)

  # Special case: input_mask will be right-aligned.
  # input_mask = jnp.einsum(
  #     "n...,nm->m...", input_mask, x_permut).astype(input_mask.dtype)
  input_mask = jnp.arange(x.shape[0]) >= (x.shape[0] - seqlen)
  return x, attn_mask, input_mask


class GemmaBlock(gemma.Block):
  """Gemma Transformer block variant which collects intermediate outputs."""

  def __call__(self, x, scan_arg, positions, attn_mask,
               decode, deterministic=True):
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    inputs_normalized = self.pre_attention_norm(x)
    attn_output = self.attn(inputs_normalized, positions, attn_mask,
                            decode, deterministic)
    attn_output = self.drop(attn_output, deterministic)
    attn_output += x
    residual = attn_output
    attn_output = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = self.drop(outputs, deterministic)
    outputs = residual + outputs
    scan_arg["outputs"] = outputs
    return outputs, scan_arg


class GemmaBackbone(nn.Module):
  """Gemma backbone (only decoder: no embedding and no logits)."""

  width: int
  depth: int
  mlp_dim: int
  num_heads: int
  num_kv_heads: int
  head_dim: int
  out_dim: int
  norm_eps: float

  dropout: float = 0.0
  dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
  cache_dtype: str | None = None

  # TODO: Wire this in all places needed so that the model can be
  # run with different activation dtype. For now only float32 runs.
  embed_dtype: str = "float32"
  head_dtype: str = "float32"

  scan: bool = False
  remat_policy: str = "none"

  @nn.compact
  def __call__(
      self, x, *,
      mask,
      positions=None, decode=False, deterministic=True,
  ):
    """Forward pass through the transformer backbone.

    Args:
      x: [B, T, E] with already embedded tokens.
      mask: Attention mask `[B, T, S]`, where its true if token T can attend to
        token S.
      positions: Optional `[B, T]` allows to specify the absolute position of
        the tokens.
      decode: Whether to use kv-cache. Caller must pass masks and positions.
      deterministic: Forwarded to all dropout layers.

    Returns:
      `(x, out)` where `x`is the output of the transformer backbone.
    """
    out = {}

    x = x.astype(self.embed_dtype)
    batch_size, seq_len, width = x.shape

    if decode:
      assert positions is not None and mask is not None, (
          "Must explicitly pass positions and mask for decoding.")

    if positions is None:
      positions = jnp.arange(seq_len).astype(jnp.int32)[None, :]
    assert positions.shape[1] == x.shape[1], (positions.shape, x.shape)

    if mask.ndim == 3:
      mask = mask[:, None, :, :]
    cache_size = max(seq_len, mask.shape[-1])
    assert mask.shape == (batch_size, 1, seq_len, cache_size), mask.shape

    if self.remat_policy == "none":
      block_cls = GemmaBlock
    else:
      block_cls = nn.remat(
          GemmaBlock,
          prevent_cse=not self.scan,
          static_argnums=(5, 6),  # 0=self, 5=decode, 6=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy),
      )

    block_kw = dict(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        embed_dim=width,
        hidden_dim=self.mlp_dim,
        dropout=self.dropout,
        dropout_bdims=self.dropout_bdims,
        cache_dtype=self.cache_dtype,
        # Gemma v1 settings:
        query_pre_attn_norm="rsqrt_head_dim",
        attn_logits_softcap=None,
        post_norms=False,
    )
    layers = self.scope.push("layers")
    if self.scan:
      blocks = [nn.scan(
          block_cls,
          # cache has axis 1 since we want leading dimension to be batch size.
          variable_axes={"params": 0, "cache": 1},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth,
      )(
          parent=layers, **block_kw
      )]
    else:
      blocks = [
          block_cls(
              parent=layers.push(str(layer)),
              **block_kw,
          )
          for layer in range(self.depth)
      ]
    scan_arg = {}
    for i, block in enumerate(blocks):
      x, scan_arg = block(
          x, scan_arg, positions, mask, decode, deterministic)
      out[f"block{i:02}_rep"] = x

    if self.scan:
      # When using scan, out only contains the final block output, so we need to
      # add the intermediate outputs from the scan_arg.
      for i, block_out in enumerate(scan_arg["outputs"]):
        out[f"block{i:02}_rep"] = block_out

    assert x.dtype == jnp.dtype(self.embed_dtype)  # Sanity check.
    out["encoded"] = x

    x = gemma.RMSNorm(name="final_norm")(x)
    out["pre_logits"] = x

    return x, out


class Model(nn.Module):
  """GIVT model supporting decoder-only applications."""
  width: int
  depth: int
  mlp_dim: int
  num_heads: int
  num_kv_heads: int
  head_dim: int
  norm_eps: float = 1e-6

  dropout: float = 0.0
  dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
  cache_dtype: str | None = None

  # TODO: Wire this in all places needed so that the model can be
  # run with different activation dtype. For now only float32 runs.
  embed_dtype: str = "float32"

  scan: bool = False
  remat_policy: str = "nothing_saveable"

  vocab_size: Optional[int] = 1000
  bos_id: Optional[int] = None
  boi_id: Optional[int] = None
  nolabel_id: Optional[int] = None
  # Multiply vocab size by this number, repeatedly embed text input with
  # different vocabs, and concatenate. This is mainly aimed at class-conditional
  # generation where the text sequence length is 1.
  num_vocab_repeats: int = 1
  causal_mask_on_prefix: bool = True
  untie_output_vocab: bool = False

  num_mixtures: int = 4
  multivariate: bool = False
  out_dim: int = 32
  scale_tol: float = 1e-6
  head_dtype: str = "float32"
  per_modality_final_norm: bool = False

  # Set to >0 for CFG support.
  drop_labels_probability: float = 0.0

  @property
  def num_logits(self) -> int:
    if self.multivariate:
      assert self.num_mixtures == 1
      # d**2 covariance, d means.
      # Note: `round` makes pytype happy.
      return round(self.out_dim ** 2) + self.out_dim
    else:
      # Mixture weights plus mean/scale per mixture
      return self.num_mixtures + 2 * self.num_mixtures * self.out_dim

  def setup(self) -> None:
    assert self.num_mixtures > 0

    if self.multivariate and self.num_mixtures != 1:
      raise ValueError("Cannot do multivariate GMM!")

    self.text_emb = nn.Embed(
        self.vocab_size * self.num_vocab_repeats,
        self.width,
        name="EmbedText",
        embedding_init=nn.initializers.normal(stddev=1.0),
    )
    if self.untie_output_vocab:
      assert self.num_vocab_repeats == 1
      self._text_logits = nn.Dense(
          self.vocab_size,
          name="LogitsText",
          kernel_init=nn.initializers.normal(stddev=1.0),
      )

    self.img_emb = nn.Dense(self.width, name="EmbedImage")
    self._img_logits = nn.Dense(
        self.num_logits,
        kernel_init=nn.initializers.zeros,
        name="LogitsImage",
        dtype=self.head_dtype,
    )

    if self.per_modality_final_norm:
      self.text_norm = gemma.RMSNorm(name="TextNorm")
      self.img_norm = gemma.RMSNorm(name="ImageNorm")

    self.decoder = GemmaBackbone(
        width=self.width,
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        out_dim=self.num_logits,
        norm_eps=self.norm_eps,
        dropout=self.dropout,
        dropout_bdims=self.dropout_bdims,
        cache_dtype=self.cache_dtype,
        embed_dtype=self.embed_dtype,
        head_dtype=self.head_dtype,
        scan=self.scan,
        remat_policy=self.remat_policy,
    )

  def prefill_cache(self, x, attn_mask, input_mask, *, cache_size):
    """Initializes decoding cache with `x` [B, N, E] and returns pre-logits."""
    # WARNING: This function right aligns the inputs! The output
    # positions do not match the input tokens for examples which have been
    # aligned to the right. OTOH x[:,-1:] will match the last prefilled token.
    _, prefill_len, _ = x.shape

    # Examples have variable number of tokens. We need to align the valid
    # tokens to the right as the cache may depend on it.
    x, attn_mask, input_mask = right_align(x, attn_mask, input_mask)
    seq_len = jnp.sum(input_mask, axis=-1)
    positions = jnp.cumsum(input_mask, axis=-1) - 1

    self.put_variable("cache", "seq_len", seq_len)
    self.put_variable("cache", "cache_begin", prefill_len - seq_len)
    self.put_variable("cache", "cache_end",
                      jnp.full(seq_len.shape, prefill_len))

    # Extend mask to cache_size.
    attn_mask = jnp.pad(
        attn_mask, ((0, 0), (0, 0), (0, cache_size - prefill_len)))

    x, out = self.decoder(x, positions=positions, mask=attn_mask, decode=True)
    if self.per_modality_final_norm:
      x = out["encoded"]
    return x

  def extend_cache(self, x):
    """Extends decoding cache with `x` [B, 1, E] and returns pre-logits."""
    assert x.shape[1] == 1, "Only supports extend the cache by one token."
    assert self.decoder.scan, "Not implemented yet."
    cache_size = self.variables[
        "cache"]["decoder"]["layers"]["attn"]["k_cache"].shape[2]

    # Lookup current token position and increment by one for next call.
    positions = self.get_variable("cache", "seq_len")
    self.put_variable("cache", "seq_len", positions + 1)

    # Update which cache positions are in use and construct attention mask.
    # Tokens can attend to all cache positions which are in use including self.
    cache_begin = self.get_variable("cache", "cache_begin")
    cache_end = self.get_variable("cache", "cache_end") + 1
    self.put_variable("cache", "cache_end", cache_end)
    mask = jnp.logical_and(
        jnp.arange(cache_size)[None, None, :] >= cache_begin[:, None, None],
        jnp.arange(cache_size)[None, None, :] < cache_end[:, None, None])

    x, out = self.decoder(
        x, positions=positions[:, None], mask=mask, decode=True)
    if self.per_modality_final_norm:
      x = out["encoded"]
    return x

  def _square_plus(self, x):
    # Via https://twitter.com/jon_barron/status/1387167648669048833
    return (x + jnp.sqrt(jnp.square(x) + 4)) / 2

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

  @staticmethod
  def get_pmf(logits: jax.Array) -> distrax.Distribution:
    return distrax.Categorical(logits=logits)

  def __call__(
      self,
      text_tokens: jax.Array,
      image_tokens: jax.Array,
      text_first_mask: jax.Array,
      *,
      text_input_mask: jax.Array | None = None,
      drop_prefix: jax.Array | None = None,
      train: bool = False,
  ) -> tuple[jax.Array, distrax.Distribution]:
    """Applies Transformer model on the inputs.

    Args:
      text_tokens: batch of text token sequences [B, L].
      image_tokens: batch of image soft token sequences [B, L, D].
      text_first_mask: batch of text masks (1=text, 0=image) [B].
      text_input_mask: boolean [B, L] indicating which are valid tokens.
      drop_prefix: If given, drop labels of the corresponding batches [B].
      train: whether it is training.

    Returns:
      Return (text_logits, image_logits, pmf, pdf, decoder_out).
    """
    x, attn_mask, input_mask = self.embed_image_and_text(
        text_tokens, image_tokens,
        text_first_mask=text_first_mask, text_input_mask=text_input_mask,
        drop_prefix=drop_prefix)

    positions = jnp.cumsum(input_mask, axis=-1) - 1
    prelogits, decoder_out = self.decoder(x, mask=attn_mask,
                                          positions=positions,
                                          deterministic=not train)
    if self.per_modality_final_norm:
      prelogits = decoder_out["encoded"]

    # Note: For now the text part never tries to predict "boi" that follows it.
    # That is not an issue for now as the text has a fixed number of tokens, for
    # variable number of tokens, one should replace "eos" with "boi" in the
    # labels for the loss of the prefix.
    text_prelogits, img_prelogits = self.split_image_and_text_prelogits(
        prelogits, text_first_mask, text_tokens.shape[1], image_tokens.shape[1],
    )

    text_logits = self.text_logits(text_prelogits)
    pmf = self.get_pmf(text_logits)
    image_logits = self.img_logits(img_prelogits)
    pdf = self.get_pdf(image_logits)
    return text_logits, image_logits, pmf, pdf, decoder_out

  def embed_image_and_text(
      self, text_tokens, image_tokens,
      *, text_first_mask, text_input_mask=None, drop_prefix=None, shift=True):
    assert text_tokens is not None and image_tokens is not None

    if text_input_mask is None:  # Default to all text tokens being valid.
      text_input_mask = jnp.full(text_tokens.shape, True)

    txt_prefix, img_prefix = text_first_mask, ~text_first_mask

    # Embed image and text tokens.
    if self.num_vocab_repeats > 1:
      offsets = jnp.repeat(
          jnp.arange(self.num_vocab_repeats) * self.vocab_size,
          text_tokens.shape[1])
      def _repeat_text(tokens):
        return jnp.tile(tokens, (1, self.num_vocab_repeats)) + offsets[None]
      nolabel = jnp.full_like(text_tokens, self.nolabel_id)
      nolabel = _repeat_text(nolabel)
      nolabel = self.text_emb(nolabel)
      text_tokens = _repeat_text(text_tokens)
      text_input_mask = jnp.tile(text_input_mask, (1, self.num_vocab_repeats))
    else:
      nolabel = self.lookup_token(self.nolabel_id, batch_size=1)
    x_txt = self.text_emb(text_tokens)
    x_img = self.img_emb(image_tokens)

    x_txt_m = text_input_mask
    x_img_m = jnp.full(x_img.shape[:-1], True)

    # Replace embedded tokens with nolabel if they were to be dropped.
    if drop_prefix is not None:
      drop_txt = txt_prefix & drop_prefix
      drop_img = img_prefix & drop_prefix
      x_txt = jnp.where(drop_txt[:, None, None], nolabel, x_txt)
      # Always use full prefix when dropping it.
      x_txt_m = jnp.where(
          drop_txt[:, None], jnp.full_like(x_txt_m, True), x_txt_m)
      x_img = jnp.where(drop_img[:, None, None], nolabel[:, :1, :], x_img)

    # There are two versions controlled by whether we have a boi token.
    batch_size, _, _ = image_tokens.shape
    if self.boi_id is not None:
      # [bos, text, boi, image], [boi, image, bos, text]
      bos = self.lookup_token(self.bos_id, batch_size)
      boi = self.lookup_token(self.boi_id, batch_size)
      bos_m = jnp.full(bos.shape[:-1], True)
      boi_m = jnp.full(boi.shape[:-1], True)
      x_txt_img = jnp.concatenate([bos, x_txt, boi, x_img], axis=1)
      x_txt_img_m = jnp.concatenate([bos_m, x_txt_m, boi_m, x_img_m], axis=1)
      x_img_txt = jnp.concatenate([boi, x_img, bos, x_txt], axis=1)
      x_img_txt_m = jnp.concatenate([boi_m, x_img_m, bos_m, x_txt_m], axis=1)

    else:
      # [bos, text, image], [bos, image, text]
      bos = self.lookup_token(self.bos_id, batch_size)
      bos_m = jnp.full(bos.shape[:-1], True)
      x_txt_img = jnp.concatenate([bos, x_txt, x_img], axis=1)
      x_txt_img_m = jnp.concatenate([bos_m, x_txt_m, x_img_m], axis=1)
      x_img_txt = jnp.concatenate([bos, x_img, x_txt], axis=1)
      x_img_txt_m = jnp.concatenate([bos_m, x_img_m, x_txt_m], axis=1)

    if shift:
      x_txt_img = x_txt_img[:, :-1]
      x_img_txt = x_img_txt[:, :-1]
      x_txt_img_m = x_txt_img_m[:, :-1]
      x_img_txt_m = x_img_txt_m[:, :-1]

    # Select between the two versions: [text,image], [image,text].
    x = jnp.where(txt_prefix[:, None, None], x_txt_img, x_img_txt)
    input_mask = jnp.where(txt_prefix[:, None], x_txt_img_m, x_img_txt_m)

    batch_size, seq_len = x.shape[:2]
    attn_mask = nn.attention.make_causal_mask(
        jnp.ones([batch_size, seq_len])).squeeze(1)
    # Optionally remove attention mask for prefix tokens.
    if not self.causal_mask_on_prefix:
      txt_prefix_mask = jnp.full_like(
          input_mask, False).at[:, :x_txt.shape[1] + 1].set(True)
      img_prefix_mask = jnp.full_like(
          input_mask, False).at[:, :x_img.shape[1] + 1].set(True)
      prefix_mask = jnp.where(
          txt_prefix[:, None], txt_prefix_mask, img_prefix_mask)
      # Set all unmasked input positions corresponding to prefix tokens to True.
      attn_mask = jnp.logical_or(attn_mask, prefix_mask[:, None, :])

    # Also mask out the attn_mask with input mask. attn_mask is [B, N1, N2]
    # and indicates if token N1 can attend to token N2. We mask the N2 part.
    attn_mask = jnp.logical_and(attn_mask, input_mask[:, None, :])

    return x, attn_mask, input_mask

  def split_image_and_text_prelogits(
      self, prelogits, text_first_mask,
      text_len, image_len):
    # There are two versions controlled by whether we have a boi token.
    if self.boi_id is not None:
      # [bos, text, boi, image], [boi, image, bos, text]
      a_txt = prelogits[:, :text_len]
      a_img = prelogits[:, self.num_vocab_repeats*text_len+1:]
      b_img = prelogits[:, :image_len]
      b_txt = prelogits[:, image_len+1:image_len+1+text_len]
    else:
      # [bos, text, image], [bos, image, text]
      a_txt = prelogits[:, :text_len]
      a_img = prelogits[:, self.num_vocab_repeats*text_len:]
      b_img = prelogits[:, :image_len]
      b_txt = prelogits[:, image_len:image_len+text_len]

    txt_prelogits = jnp.where(text_first_mask[:, None, None], a_txt, b_txt)
    img_prelogits = jnp.where(text_first_mask[:, None, None], a_img, b_img)
    return txt_prelogits, img_prelogits

  def lookup_token(self, token_id: int, batch_size: int):
    """Lookup a statically defined token (e.g. bos, boi, nolabel)."""
    assert isinstance(token_id, int)
    # TODO: Avoid matmul with full matrix for this code path.
    return jnp.repeat(
        self.text_emb(jnp.full((1, 1), token_id)), batch_size, axis=0)

  def text_logits(self, pre_logits):
    if self.per_modality_final_norm:
      pre_logits = self.text_norm(pre_logits)
    if self.untie_output_vocab:
      return self._text_logits(pre_logits)
    return self.text_emb.attend(pre_logits)

  def img_logits(self, pre_logits):
    if self.per_modality_final_norm:
      pre_logits = self.img_norm(pre_logits)
    return self._img_logits(pre_logits)

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
) -> Any:
  """Loads params from init checkpoint and merges into init_params."""
  del model_params
  assert isinstance(init_files, str), init_files
  ckpt_params = utils.load_params(init_files)
  ckpt_params = common.merge_params(ckpt_params, init_params, dont_load)

  return ckpt_params
