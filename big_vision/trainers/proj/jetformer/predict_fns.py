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

"""Prediction functions for JetFormer."""
# pylint: disable=consider-using-from-import
from absl import logging
import big_vision.models.proj.givt.parallel_decode as parallel_decode
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


# Utils to encode and decode images as latents.
def encode_images(  # pylint: disable=missing-function-docstring
    params, images, *, adaptor, patch_pca, rngs, reparametrize: bool):
  # Apply patch_pca module.
  x, logvar = patch_pca.apply({}, images, method=patch_pca.encode, rngs=rngs)
  if reparametrize:
    x = patch_pca.apply(
        {}, x, logvar, method=patch_pca.reparametrize, rngs=rngs)

  # Apply invertible network.
  if adaptor is not None:
    x = unflatten_latents(x)
    x, _ = adaptor.apply({"params": params}, x, method=adaptor.forward)
    x = flatten_latents(x)
  return x


def decode_images(params, x, *, adaptor, patch_pca):
  # Apply invertible network backwards.
  if adaptor is not None:
    x = unflatten_latents(x)
    x, _ = adaptor.apply({"params": params}, x, method=adaptor.inverse)
    x = flatten_latents(x)

  # Apply patch_pca module backwards.
  images = patch_pca.apply({"params": {}}, x, method=patch_pca.decode)
  return images


def unflatten_latents(x):
  hw = int(x.shape[1] ** 0.5)
  return einops.rearrange(x, "b (h w) c -> b h w c", h=hw, w=hw)


def flatten_latents(x):
  return einops.rearrange(x, "b h w c -> b (h w) c")


# Utils to sample the decoder.
def sample_image_latents(
    params, batch, *, model,
    decode_len=256, temperature=1.0, temperature_probs=1.0,
    cfg_weight=None, rng=None):
  """Sample image latents conditioned on text prompt."""
  rng = rng if rng is not None else jax.random.PRNGKey(
      jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))
  # The following makes sure to skip CFG if cfg_weight is a Python float or int
  # and is 0, or is None, but performs CFG if cfg_weight is a traced array.
  if isinstance(cfg_weight, (int, float)):
    do_cfg = (cfg_weight != 0)
  else:
    do_cfg = cfg_weight is not None
  logging.info("Sampling with cfg_weight=%r", cfg_weight)

  def _sample_prelogits(model, pre_logits):
    rng = model.make_rng("sample")
    logits = model.img_logits(pre_logits)
    if do_cfg:
      # get_pdf is not jax-friendly as it returns an opaque object so we have
      # to split logits before calling it.
      logits_cond, logits_uncond = einops.rearrange(
          logits, "(b s) ... -> s b ...", s=2)
      pdf_cond = model.get_pdf(logits_cond,
                               temperature_scales=temperature,
                               temperature_probs=temperature_probs)
      pdf_uncond = model.get_pdf(logits_uncond,
                                 temperature_scales=temperature,
                                 temperature_probs=temperature_probs)
      pdf = parallel_decode.CFGDensity(
          pdf_cond, pdf_uncond,
          w=cfg_weight, rng=rng)
      samples = pdf.sample(seed=rng)
      logprobs = pdf.log_prob(samples)
      logprobs = jnp.sum(logprobs, axis=2)  # [B, N, C] -> [B, N]
      return jnp.repeat(samples, 2, axis=0), jnp.repeat(logprobs, 2, axis=0)
    else:
      pdf = model.get_pdf(logits,
                          temperature_scales=temperature,
                          temperature_probs=temperature_probs)
      samples = pdf.sample(seed=rng)
      logprobs = pdf.log_prob(samples)
      return samples, logprobs

  # Main sample logic where "model" has been bound.
  def _sample(model):
    text = batch["text"]
    text_mask = batch.get("text_mask", jnp.full(text.shape, True))

    # Add unconditional sequences if needed [x, x_uncond, y, y_uncond, ...]
    if do_cfg:
      drop_prefix = jnp.tile(jnp.array([False, True]), text.shape[0])
      text = jnp.repeat(text, 2, axis=0)
      # Overridden to full mask when dropping the label in embed_image_and_text.
      text_mask = jnp.repeat(text_mask, 2, axis=0)
    else:
      drop_prefix = None

    # Prepare inputs to prefill the decoder. Pass images of seq_len=0 so it
    # prefills up to the BOI of images.
    batch_size, _ = text.shape
    images = jnp.zeros((batch_size, 0, model.out_dim))  # zero-len images.
    text_first_mask = jnp.full((batch_size,), True)
    x, attn_mask, input_mask = model.embed_image_and_text(
        text, images,
        text_first_mask=text_first_mask,
        text_input_mask=text_mask,
        drop_prefix=drop_prefix, shift=False)

    # Prefill the decoder cache with 'x' and sample the first output.
    cache_size = x.shape[1] + decode_len - 1
    last_prelogits = model.prefill_cache(x, attn_mask, input_mask,
                                         cache_size=cache_size)[:, -1:]
    tokens, logp = _sample_prelogits(model, last_prelogits)

    # Init loop state with the token decoded during prefill.
    batch_size, _, prelogits_dim = last_prelogits.shape
    out_prelogits = jnp.zeros((batch_size, decode_len, prelogits_dim))
    out_logp = jnp.zeros((batch_size, decode_len))
    out_tokens = jnp.zeros((batch_size, decode_len, tokens.shape[2]))

    out_prelogits = out_prelogits.at[:, 0:1].set(last_prelogits)
    out_tokens = out_tokens.at[:, 0:1].set(tokens)
    out_logp = out_logp.at[:, 0:1].set(logp)

    # Most callers will only need the out_tokens (i.e. the sample latents).
    # This code pattern allows one to easily add other outputs if needed.
    # last_tokens is a carry variable to avoid a dynamic lookup in the loop.
    state = {
        "out_prelogits": out_prelogits,  # [B, decode_len, D]
        "out_tokens": out_tokens,        # [B, decode_len, H]
        "out_logp": out_logp,            # [B, decode_len]
        "last_tokens": tokens,           # [B, 1, H]
    }

    # Loop to decode remaining tokens. This function will be called by nn.scan
    # with xs = 1, 2, ..., decode_len and update the state accordingly.
    def loop_step(model, state, xs):
      x = model.img_emb(state["last_tokens"])
      prelogits = model.extend_cache(x)
      tokens, logp = _sample_prelogits(model, prelogits)
      # Update state with the new tokens.
      state["out_prelogits"] = jax.lax.dynamic_update_slice(
          state["out_prelogits"], prelogits, (0, xs, 0))
      state["out_tokens"] = jax.lax.dynamic_update_slice(
          state["out_tokens"], tokens, (0, xs, 0))
      state["out_logp"] = jax.lax.dynamic_update_slice(
          state["out_logp"], logp, (0, xs))
      state["last_tokens"] = tokens
      return state, None
    # Note that besides "state", the "cache" variables are also carried and
    # that the sample rng state is splitted so each call sees a different rng.
    xs = jnp.arange(1, decode_len)
    state, _ = nn.scan(
        loop_step, variable_broadcast="params", variable_carry="cache",
        split_rngs={"sample": True})(model, state, xs)
    del state["last_tokens"]  # Not needed anymore.

    # Remove unconditional sequences if needed.
    if do_cfg:
      state = jax.tree.map(lambda x: x[::2], state)

    return state

  out, _ = nn.apply(_sample, model, mutable=["cache"])(
      {"params": params}, rngs={"sample": rng})

  return out


def sample_text(
    params, batch, *, model,
    decode_len=64, temperature=1.0, rng=None):
  """Sample text continuation conditioned on image."""
  rng = rng if rng is not None else jax.random.PRNGKey(
      jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))

  def _sample_prelogits(model, pre_logits):
    rng = model.make_rng("sample")
    logits = model.text_logits(pre_logits)
    # Sample using temperature.
    # TODO: Add top-k/top-p/etc...
    modified_pmf = model.get_pmf(logits / temperature)
    samples = modified_pmf.sample(seed=rng)
    # Return logp according to the original unmodified distribution.
    pmf = model.get_pmf(logits)
    logprobs = pmf.log_prob(samples)
    return samples, logprobs

  # Main sample logic where "model" has been bound.
  def _sample(model):
    images = batch["image_latents"]

    # TODO: Add support for CFG.
    drop_prefix = None

    # Prepare inputs to prefill the decoder with [image, Optional[text]].
    batch_size, _, _ = images.shape
    text_first_mask = jnp.full((batch_size,), False)

    if batch["text"] is None:
      # Zero-len text, so it prefills up to the BOS of text.
      text = jnp.full((batch_size, 0), 0)
      text_input_mask = jnp.full((batch_size, 0), True)
    else:
      # If text is present, it will prefill the BOS of text and the tokens
      # which are true in the text_mask (i.e. each example can have a variable
      # number of text tokens prefilled).
      text = batch["text"]
      text_input_mask = batch["text_mask"]

    x, attn_mask, input_mask = model.embed_image_and_text(
        text, images,
        text_first_mask=text_first_mask,
        text_input_mask=text_input_mask,
        drop_prefix=drop_prefix, shift=False)

    # Prefill the decoder cache with 'x' and sample the first output.
    cache_size = x.shape[1] + decode_len - 1
    last_prelogits = model.prefill_cache(
        x, attn_mask, input_mask, cache_size=cache_size)[:, -1:]
    tokens, logp = _sample_prelogits(model, last_prelogits)

    # Init loop state with the token decoded during prefill.
    batch_size, _, prelogits_dim = last_prelogits.shape
    out_prelogits = jnp.zeros((batch_size, decode_len, prelogits_dim),
                              dtype=last_prelogits.dtype)
    out_logp = jnp.zeros((batch_size, decode_len), dtype=logp.dtype)
    out_tokens = jnp.zeros((batch_size, decode_len), dtype=tokens.dtype)

    out_prelogits = out_prelogits.at[:, 0:1].set(last_prelogits)
    out_tokens = out_tokens.at[:, 0:1].set(tokens)
    out_logp = out_logp.at[:, 0:1].set(logp)

    # Most callers will only need the out_tokens (i.e. the sample latents).
    # This code pattern allows one to easily add other outputs if needed.
    # last_tokens is a carry variable to avoid a dynamic lookup in the loop.
    state = {
        "out_prelogits": out_prelogits,  # [B, decode_len, D]
        "out_tokens": out_tokens,        # [B, decode_len]
        "out_logp": out_logp,            # [B, decode_len]
        "last_tokens": tokens,           # [B, 1]
    }

    # Loop to decode remaining tokens. This function will be called by nn.scan
    # with xs = 1, 2, ..., decode_len and update the state accordingly.
    def loop_step(model, state, xs):
      x = model.text_emb(state["last_tokens"])
      prelogits = model.extend_cache(x)
      tokens, logp = _sample_prelogits(model, prelogits)
      # Update state with the new tokens.
      state["out_prelogits"] = jax.lax.dynamic_update_slice(
          state["out_prelogits"], prelogits, (0, xs, 0))
      state["out_tokens"] = jax.lax.dynamic_update_slice(
          state["out_tokens"], tokens, (0, xs))
      state["out_logp"] = jax.lax.dynamic_update_slice(
          state["out_logp"], logp, (0, xs))
      state["last_tokens"] = tokens
      return state, None
    # Note that besides "state", the "cache" variables are also carried and
    # that the sample rng state is splitted so each call sees a different rng.
    xs = jnp.arange(1, decode_len)
    state, _ = nn.scan(
        loop_step, variable_broadcast="params", variable_carry="cache",
        split_rngs={"sample": True})(model, state, xs)
    del state["last_tokens"]  # Not needed anymore.

    return state

  out, _ = nn.apply(_sample, model, mutable=["cache"])(
      {"params": params}, rngs={"sample": rng})

  return out
