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

"""Image encoder + AR-decoder LLM."""

import importlib
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

ConfigDict = Any


def make_attn_mask(input_mask, mask_ar):
  """Returns attention mask bool[B, N, N] to use in transformer.

  Tokens can attend to valid inputs tokens which have a cumulative mask_ar
  smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
  setup several types of attention, for example:

    [[1 1 1 1 1 1]]: pure causal attention.

    [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
        themselves and the last 3 tokens have a causal attention. The first
        entry could also be a 1 without changing behaviour.

    [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
        block can attend all previous blocks and all tokens on the same block.

  Args:
    input_mask: bool[B, N] true if its part of the input, false if padding.
    mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
      it and 0 where it shares the same attention mask as the previous token.
  """
  cumsum = jnp.cumsum(mask_ar, axis=1)
  attn_mask = (cumsum[:, None, :] <= cumsum[:, :, None])
  valid_mask = (input_mask[:, None, :] * input_mask[:, :, None])
  return jnp.logical_and(attn_mask, valid_mask)


class Model(nn.Module):
  """Two towers transformer."""
  img_model: str = "vit"
  img: Optional[ConfigDict] = None
  llm_model: str = "proj.paligemma.gemma_bv"
  llm: Optional[ConfigDict] = None

  def setup(self):
    self._llm = importlib.import_module(
        f"big_vision.models.{self.llm_model}"
    ).Model(**(self.llm or {}), name="llm")

    img_config = {"num_classes": self._llm.embdim, **(self.img or {})}
    self._img_model = importlib.import_module(
        f"big_vision.models.{self.img_model}"
    ).Model(**img_config, name="img")

  def embed_image(self, image, train=False):
    out = {}

    # if we have video, fold frame dimension into the batch dimension
    image_shape = image.shape
    if len(image_shape) == 5:  # video frames
      image = jnp.reshape(image, (-1, *image.shape[-3:]))

    # Do we want to normalize? are they huge?
    zimg, out_img = self._img_model(image, train=train)

    if len(image_shape) == 5:  # concatenate tokens from all video frames
      zimg = jnp.reshape(zimg, (image_shape[0], -1, zimg.shape[-1]))

    out["img/zimg"] = zimg
    for k, v in out_img.items():
      out[f"img/{k}"] = v
    return zimg, out

  def embed_text(self, tokens, train=False):
    out = {}
    ztxt = out["llm/ztxt"] = self._llm.embed_tokens(tokens, train=train)
    return ztxt, out

  def embed_image_and_text(self, image, text, *,
                           input_mask=None, mask_ar=None, train=False):
    """Concats image/text into a sequence of embeded tokens to pass to `llm`.

    Args:
      image: float[B, H, W, 3] image to be embedded by the `img` model and used
        as prefix to the sequence passed to the `llm` model.
      text: int32[B, T] token sequence to embedded by the `llm`.
      input_mask: bool[B, T] true if the text token is a valid token and false
        if its a token to pad the sequence. Defaults to all being input tokens.
      mask_ar: int32[B, T] mask that's 1 where `text` should be attended to
        causally, and 0 where it can be attended to with full self-attention.
        Defaults to all text tokens being auto-regressive.
      train: bool whether we're in train or test mode (dropout etc).

    Returns:
      Tuple (x: float[B, N, E], input_mask: bool[B, N], mask_ar: int[B, N]) and
      auxiliary outputs.
    """
    zimg, out_img = self.embed_image(image, train=train)
    ztxt, out_txt = self.embed_text(text, train=train)

    if input_mask is None:
      input_mask = jnp.full(text.shape, True)
    if mask_ar is None:
      mask_ar = jnp.full(text.shape, 1)

    # Concatenate embeded image and text into a single token sequence.
    x = jnp.concatenate([zimg, ztxt], axis=1)
    _, img_len, _ = zimg.shape
    pad_width = ((0, 0), (img_len, 0))
    mask_ar = jnp.pad(mask_ar, pad_width, constant_values=0)
    input_mask = jnp.pad(input_mask, pad_width, constant_values=True)

    return (x, input_mask, mask_ar), {**out_img, **out_txt}

  def __call__(self, image, text, mask_ar, train=False):
    """Concats image/text and returns text logits.

    Args:
      image: float32[B, H, W, 3] image that can be passed to the `img` model.
      text: int32[B, T] token sequence that can be embedded by the `txt` model.
      mask_ar: int32[B, T] mask that's 1 where `text` should be attended to
        causally, and 0 where it can be attended to with full self-attention.
      train: bool whether we're in train or test mode (dropout etc).

    Returns:
      float32[B, T, V] logits for the `text` input, and an out-dict of named
      intermediates.
    """
    # Embed the image and text.
    (x, input_mask, mask_ar), out = self.embed_image_and_text(
        image, text, mask_ar=mask_ar, train=train)

    # Call transformer on the embedded token sequence.
    attn_mask = out["attn_mask"] = make_attn_mask(input_mask, mask_ar)
    _, out_llm = self._llm(x, mask=attn_mask, train=train)
    for k, v in out_llm.items():
      out[f"llm/{k}"] = v

    # Extract the logits for the text tokens.
    zimg = out["img/zimg"]
    text_pre_logits = out["llm/pre_logits"][:, zimg.shape[1]:, :]
    text_logits = self._llm.compute_logits(text_pre_logits, train=train)
    out["text_logits"] = text_logits
    out["text_tokens"] = jnp.argmax(text_logits, axis=-1)
    return text_logits, out

  def prefill_cache(self, x, input_mask, mask_ar, *, cache_size):
    """Initializes decoding cache with `x` [B, N, E] as prompt."""
    if hasattr(self._llm, "prefill_cache"):
      attn_mask = make_attn_mask(input_mask, mask_ar)
      return self._llm.prefill_cache(
          x, input_mask, attn_mask, cache_size=cache_size)
    else:
      return self._fallback_prefill_cache(x, input_mask, mask_ar, cache_size)

  def extend_cache(self, x):
    """Advances decoding cache with `x` [B, 1, E]."""
    if hasattr(self._llm, "prefill_cache"):
      return self._llm.extend_cache(x)
    else:
      return self._fallback_extend_cache(x)

  def _fallback_prefill_cache(self, x, input_mask, mask_ar, cache_size):
    # FALLBACK: only cache inputs and call the model with the full sequence
    # for each and every decode step. Very slowwww...
    #
    # This very slow codepath does not requires the model to implement caching.
    # It is intended to allow to plug any model under development quite early
    # into some decoding tasks and not as a long term decoding solution.
    attn_mask = make_attn_mask(input_mask, mask_ar)
    logits, _ = self._llm(x, mask=attn_mask)

    # Save the prefill inputs for subsequent extend_calls in the cache.
    # Unused entries are zero-initialized.
    pad_size = cache_size - x.shape[1]
    x = jnp.pad(jnp.where(input_mask[..., None], x, 0),
                [(0, 0), (0, pad_size), (0, 0)])
    mask_ar = jnp.pad(jnp.where(input_mask, mask_ar, 0),
                      [(0, 0), (0, pad_size)])
    input_mask = jnp.pad(input_mask, [(0, 0), (0, pad_size)])
    self.put_variable("cache", "x_cache", x)
    self.put_variable("cache", "input_mask_cache", input_mask)
    self.put_variable("cache", "mask_ar_cache", mask_ar)

    # Extract logits of the last token (using einsum).
    last_pos = jnp.sum(input_mask, axis=1)[:, None] - 1
    last_onehot = jax.nn.one_hot(last_pos, logits.shape[1], dtype=jnp.int32)
    last_logits = jnp.einsum("bnh,ben->beh", logits, last_onehot)

    return last_logits

  def _fallback_extend_cache(self, x):
    # FALLBACK: append inputs to cache and call the model with the full sequence
    # for each and every decode step. Very slowwww...
    assert x.shape[1] == 1
    mask_ar = jnp.full(x.shape[:-1], 1)
    input_mask = jnp.full(x.shape[:-1], True)

    # Append inputs to cache by add/or on the next available cache position,
    # which is zero-initialized.
    c_x = self.get_variable("cache", "x_cache")
    c_input_mask = self.get_variable("cache", "input_mask_cache")
    c_mask_ar = self.get_variable("cache", "mask_ar_cache")
    next_pos = jnp.sum(c_input_mask, axis=1)[:, None]
    move_onehot = jax.nn.one_hot(next_pos, c_x.shape[1], dtype=jnp.int32)
    x = jnp.add(c_x, jnp.einsum("beh,ben->bnh", x, move_onehot))
    mask_ar = jnp.add(c_mask_ar, jnp.einsum("be,ben->bn", mask_ar, move_onehot))
    input_mask = jnp.logical_or(
        c_input_mask, jnp.einsum("be,ben->bn", input_mask, move_onehot))
    self.put_variable("cache", "x_cache", x)
    self.put_variable("cache", "input_mask_cache", input_mask)
    self.put_variable("cache", "mask_ar_cache", mask_ar)

    # Call model on the full cached sequence.
    attn_mask = make_attn_mask(input_mask, mask_ar)
    logits, _ = self._llm(x, mask=attn_mask)

    # Extract logits of the last token.
    last_pos = jnp.sum(input_mask, axis=1)[:, None] - 1
    last_onehot = jax.nn.one_hot(last_pos, logits.shape[1], dtype=jnp.int32)
    last_logits = jnp.einsum("bnh,ben->beh", logits, last_onehot)

    return last_logits


# pylint: disable=line-too-long
import os
GEMMA_DIR = os.environ.get("BV_GEMMA_DIR", "PLEASE_SET_BV_GEMMA_DIR")
VANITY_NAMES = {
    # Because checkpoints are behind an ACK-wall, the user has to download them
    # to some folder (or bucket), take that from an environment variable.
    "pt_224": os.path.join(GEMMA_DIR, "pt_224.npz"),
    "pt_224.bf16": os.path.join(GEMMA_DIR, "pt_224.bf16.npz"),
    "pt_224.f16": os.path.join(GEMMA_DIR, "pt_224.f16.npz"),
    "pt_448": os.path.join(GEMMA_DIR, "pt_448.npz"),
    "pt_448.bf16": os.path.join(GEMMA_DIR, "pt_448.bf16.npz"),
    "pt_448.f16": os.path.join(GEMMA_DIR, "pt_448.f16.npz"),
    "pt_896": os.path.join(GEMMA_DIR, "pt_896.npz"),
    "pt_896.bf16": os.path.join(GEMMA_DIR, "pt_896.bf16.npz"),
    "pt_896.f16": os.path.join(GEMMA_DIR, "pt_896.f16.npz"),
}
# pylint: enable=line-too-long


def load(init_params, init_files, model_cfg, img_load_kw={}, llm_load_kw={}):  # pylint: disable=dangerous-default-value
  """Loads both pieces, `init_files` is now a dict with `img` and `llm` keys."""

  # A slight shortcut when loading an already combined model:
  if isinstance(init_files, str):
    init_files = VANITY_NAMES.get(init_files, init_files)
    init_files = {"img": f"{init_files}:img", "llm": f"{init_files}:llm"}

  if not init_params:  # Convenience to skip checks in colab.
    init_params = {"img": None, "llm": None}
  restored_params = {**init_params}

  init_files = {**init_files}  # Needed because ConfigDict but we'll pop stuff.

  if img_init := init_files.pop("img", None):
    restored_params["img"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('img_model', 'vit')}"
    ).load(init_params["img"], img_init, model_cfg.img, **img_load_kw)

  if llm_init := init_files.pop("llm", None):
    restored_params["llm"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('llm_model', 'proj.paligemma.gemma_bv')}"
    ).load(init_params["llm"], llm_init, model_cfg.llm, **llm_load_kw)

  assert not init_files, (
      f"There's something unused left in `config.model_init`. You probably got "
      f"a typo. Here it is: {init_files}")

  return restored_params
