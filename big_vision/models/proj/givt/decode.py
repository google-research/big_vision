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

"""Autorgregressive sampler for GIVT."""

import functools
from typing import Any, Optional

from big_vision.models.proj.givt import parallel_decode
import flax
from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
import ml_collections


def _sample_gmm(
    gmm_pdf,
    *,
    rng,
    cfg_inference_weight=None,
    gmm_pdf_uncond=None,
):
  """Draw a single sample from a GMM."""
  if cfg_inference_weight is not None:
    assert gmm_pdf_uncond is not None
    gmm_pdf = parallel_decode.CFGDensity(
        gmm_pdf, gmm_pdf_uncond, w=cfg_inference_weight, rng=rng
    )
  samples = gmm_pdf.sample(seed=rng)
  logprobs = gmm_pdf.log_prob(samples)
  if logprobs.ndim == 2:
    logprobs = logprobs[..., None]
  return samples, logprobs


# Beam search reshaping utils
def _flatten_samples_dim(x):
  """Flattens samples dimension into batch dimension."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def _unflatten_samples_dim(x, batch_size, num_samples):
  """Unflattens first dimension into batch and samples dimensions."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  assert batch_size * num_samples == x.shape[0]
  return x.reshape((batch_size, num_samples) + x.shape[1:])


def _cache_map(fn, cache, scan=False):
  """Maps function over cache."""
  if scan:
    # Assuming the chache is scanned over the first dimension, we apply a map
    # function over this dimension for scanned models
    fn_mod = lambda x: jax.lax.map(fn, x) if x.ndim > 0 else fn(x)
  else:
    fn_mod = fn

  frozen = isinstance(cache, flax.core.FrozenDict)
  if frozen:
    cache = flax.core.unfreeze(cache)
  flat_cache = flax.traverse_util.flatten_dict(cache)
  # Exclude cached relative position bias from beam expansion, etc.
  keyvals = {k: v for k, v in flat_cache.items() if k[-1] != "cached_bias"}
  keyvals = jax.tree_map(fn_mod, keyvals)
  flat_cache.update(keyvals)
  new_cache = flax.traverse_util.unflatten_dict(flat_cache)
  if frozen:
    new_cache = flax.core.freeze(new_cache)
  return new_cache


@flax.struct.dataclass
class LoopState:
  """Internal state of the sampling loop."""
  # Terminology
  # b:  batch size
  # nb: number of beams
  # nf: number of fans
  # s:  seaquence length
  # d:  feature dimension
  rng: jnp.ndarray        # PRNGKey of the loop state.
  cache: Any              # Cache for fast auto-regressive decoding.
  sequences: jnp.ndarray  # (b * nb, s, d)
  logprobs: jnp.ndarray   # (b * nb, s, d)
  cache_u: Any            # Uncond cache if cfg, otherwise None


def _create_cache(
    labels,
    model,
    init_sequence,
    params,
    encoded,
    uncond=False,
):
  """Creates the cache and returns initial logits."""
  if uncond:
    assert labels is not None  # Need labels for CFG!
    drop_labels = jnp.ones((labels.shape[0],), dtype=jnp.bool_)
  else:
    drop_labels = None

  def init_cache(model):
    return model.decode(
        init_sequence, labels, encoded, decode=True, drop_labels=drop_labels
    )

  cache = nn.apply(init_cache, model, mutable=True)(params)[1]["cache"]

  def prefill_cache(model):
    return model.prefill(
        labels, init_sequence.shape[0], encoded, drop_labels=drop_labels
    )

  # prefill class label or BOS token
  prefill_logits, aux = nn.apply(prefill_cache, model, mutable=True)(
      {"params": params["params"], "cache": cache})
  cache = aux["cache"]
  return cache, prefill_logits


def generate(
    params: Any,
    seed: jax.Array,
    *,
    model: nn.Module,
    seq_len: int,
    feature_dim: int,
    labels: Optional[jnp.ndarray] = None,
    cond_image: Optional[jnp.ndarray] = None,
    batch_size: Optional[int] = None,
    config: Optional[ml_collections.ConfigDict] = None,
) -> tuple[jax.Array, jax.Array]:
  """Sampling loop for GIVT."""
  if model.style != "ar":  # pytype: disable=wrong-arg-types
    raise ValueError(f"Invalid style: {model.style}")
  if model.has_encoder != (cond_image is not None):
    raise ValueError("Need cond_image if and only if the model has an encoder!")

  assert labels is not None or batch_size, (
      "Please provide either labels or batch_size.")

  config = config or {}
  config = dict(config)  # copy

  # For sampling, we support keep_gt (a bool mask), and gt (ground truth)
  # tokens to use instead of samples.
  keep_gt = config.pop("keep_gt", None)
  gt = config.pop("gt", None)

  if isinstance(seed, int):
    seed = jax.random.PRNGKey(seed)

  beam_size = config.pop("beam_size", 1)
  fan_size = config.pop("fan_size", 1)

  if labels is not None:
    batch_size = labels.shape[0]
    # fold beams into batch dimension
    labels = labels.repeat(beam_size, axis=0)

  # initialize sequence and logprobs (we track per feature dim logprobs)
  init_sequence = jnp.zeros((batch_size * beam_size, seq_len, feature_dim))
  init_logprobs = jnp.zeros_like(init_sequence)

  if cond_image is not None:
    # embed conditioning image if provided
    def encode_cond_img(model, cond_img):
      return model.encode(cond_img)
    encoded = nn.apply(encode_cond_img, model)(params, cond_image)
    encoded = jnp.repeat(encoded, beam_size, axis=0)
  else:
    encoded = None

  cache, prefill_logits = _create_cache(
      labels, model, init_sequence, params, encoded
  )

  cfg_inference_weight = config.pop("cfg_inference_weight", None)
  if cfg_inference_weight == 0.0:
    cfg_inference_weight = None
  cfg = cfg_inference_weight is not None

  get_pdf = functools.partial(
      model.get_pdf,
      temperature_scales=config.pop("temp", None),
      temperature_probs=config.pop("temp_probs", None),
  )

  # setup sampling function
  sample = functools.partial(
      _sample_gmm, cfg_inference_weight=cfg_inference_weight
  )

  # draw first output token
  pdf_first = get_pdf(prefill_logits)
  rng_first, rng = jax.random.split(seed)

  if cfg:
    assert beam_size == 1 and fan_size == 1  # CFG + Beam not supported.
    cache_u, prefill_logits_u = _create_cache(
        labels, model, init_sequence, params, encoded, uncond=True
    )
    pdf_first_u = get_pdf(prefill_logits_u)
  else:
    cache_u = None
    pdf_first_u = None

  tokens_first, logprobs_first = sample(
      pdf_first, rng=rng_first, gmm_pdf_uncond=pdf_first_u
  )
  init_sequence = init_sequence.at[:, 0].set(tokens_first.squeeze(axis=1))
  init_logprobs = init_logprobs.at[:, 0].set(logprobs_first.squeeze(axis=1))

  def tokens_to_logits(tokens, cache, uncond=False):
    if uncond:
      drop_labels = jnp.ones((labels.shape[0],), dtype=jnp.bool_)
    else:
      drop_labels = None

    def decode_step(model, tokens):
      return model.decode(tokens, labels, encoded,
                          decode=True, drop_labels=drop_labels)

    logits, aux = nn.apply(decode_step, model, mutable=True)(
        {"params": params["params"], "cache": cache}, tokens)
    return logits, aux["cache"]

  init_state = LoopState(
      cache=cache,
      sequences=init_sequence,  # (b * nb, s, d)
      logprobs=init_logprobs,   # (b * nb, s, d)
      rng=rng,
      cache_u=cache_u,
  )

  rand_top_k = config.pop("rand_top_k", False)
  rand_top_k_temp = config.pop("rand_top_k_temp", 1.0)

  assert not config, f"Sampling config is expected to be empty: {config}"

  def sampling_iteration(i, state):
    rng_sampling, rng_local = jax.random.split(state.rng)
    cur_tokens = state.sequences[:, i][:, None]
    # (b * nb, d)
    cur_logits, cache = tokens_to_logits(cur_tokens, state.cache)

    # (b, nb, d)
    cur_logits = _unflatten_samples_dim(
        cur_logits, batch_size, beam_size).squeeze(axis=2)

    # (b, nb * nf, d)
    cur_pdf = get_pdf(cur_logits.repeat(fan_size, axis=1))

    if cfg:
      cur_logits_u, cache_u = tokens_to_logits(
          cur_tokens, state.cache_u, uncond=True
      )
      cur_logits_u = _unflatten_samples_dim(
          cur_logits_u, batch_size, beam_size).squeeze(axis=2)
      cur_pdf_u = get_pdf(cur_logits_u.repeat(fan_size, axis=1))
      new_tokens, new_logprobs = sample(
          cur_pdf, rng=rng_sampling, gmm_pdf_uncond=cur_pdf_u
      )
    else:
      new_tokens, new_logprobs = sample(cur_pdf, rng=rng_sampling)
      cache_u = None

    if gt is not None:
      assert keep_gt is not None
      new_tokens = jnp.where(keep_gt[i], gt[:, i, :][:, None], new_tokens)

    # Skip beam search if not needed
    if beam_size == fan_size == 1:
      sampled_tokens = new_tokens.squeeze(axis=1)
      sequences = state.sequences.at[:, i + 1].set(sampled_tokens)
      return LoopState(
          cache=cache,
          rng=rng_local,
          sequences=sequences,
          logprobs=state.logprobs,
          cache_u=cache_u,
      )

    # (b, nb, s, d)
    logprobs = _unflatten_samples_dim(state.logprobs, batch_size, beam_size)
    cur_logprobs = logprobs[:, :, i]  # (b, nb, d)
    # (b, nb * nf, d)
    new_logprobs = new_logprobs + cur_logprobs.repeat(fan_size, axis=1)
    beam_logprobs = new_logprobs.sum(axis=-1)  # (b, nb * nf)

    if rand_top_k:
      # randomize top-k sampling via sampling from a categorical distribution
      def stoc_top_k(r, x, p):
        return jax.random.choice(r, x, shape=(beam_size,), replace=False, p=p)
      # construct index grid
      index_grid = jnp.arange(beam_logprobs.shape[1], dtype=jnp.int32)
      # (b, nb * nf)
      index_grid = index_grid[None].repeat(beam_logprobs.shape[0], axis=0)
      top_k_rng, rng_local = jax.random.split(rng_local)
      top_k_rng = jax.random.split(top_k_rng, beam_logprobs.shape[0])
      # vmap categorical sampling
      top_beam_fan_indices = jax.vmap(stoc_top_k, in_axes=(0, 0, 0))(
          top_k_rng,
          index_grid,
          nn.softmax(beam_logprobs / rand_top_k_temp, axis=-1))
    else:
      _, top_beam_fan_indices = lax.top_k(beam_logprobs, k=beam_size)  # (b, nb)

    top_beam_indices = top_beam_fan_indices // fan_size

    def _gather_beams(x):
      if x.ndim == 0:
        return x
      # checkify.check(jnp.all(top_beam_indices < x.shape[1]),
      #                f"`take_along_axis` out of bounds in `_gather_beams`: "
      #                f"{top_beam_indices.max()} vs. {x.shape[1]}")
      # (b, nb, 1 ... 1)
      expanded_indices = top_beam_indices.reshape(
          top_beam_indices.shape + (1,) * (x.ndim - 2))
      return jnp.take_along_axis(x, expanded_indices, axis=1)

    def _gather_tokens(x):
      # (b, nb * nf, d) -> (b, nb, d)
      # checkify.check(jnp.all(top_beam_fan_indices < x.shape[1]),
      #                f"`take_along_axis` out of bounds in `_gather_tokens`: "
      #                f"{top_beam_fan_indices.max()} vs. {x.shape[1]}")
      return jnp.take_along_axis(x, top_beam_fan_indices[..., None], axis=1)
    # (b, nb, s, d)
    sequences = _unflatten_samples_dim(state.sequences, batch_size, beam_size)
    sequences = _gather_beams(sequences)  # (b, nb, s, d)
    sequences = sequences.at[:, :, i + 1].set(_gather_tokens(new_tokens))
    # (b, nb, s, d)
    sequences = _flatten_samples_dim(sequences)

    logprobs = _gather_beams(logprobs)
    logprobs = logprobs.at[:, :, i + 1].set(_gather_tokens(new_logprobs))
    logprobs = _flatten_samples_dim(logprobs)

    scanned_cache = getattr(model, "scan", False)
    cache = _cache_map(
        lambda x: _unflatten_samples_dim(x, batch_size, beam_size),
        cache, scanned_cache)
    cache = _cache_map(_gather_beams, cache, scanned_cache)
    cache = _cache_map(_flatten_samples_dim, cache, scanned_cache)

    if cfg:
      assert cache_u is not None
      cache_u = _cache_map(
          lambda x: _unflatten_samples_dim(x, batch_size, beam_size),
          cache_u, scanned_cache
      )
      cache_u = _cache_map(_gather_beams, cache_u, scanned_cache)
      cache_u = _cache_map(_flatten_samples_dim, cache_u, scanned_cache)
    else:
      assert cache_u is None

    return LoopState(
        cache=cache,
        rng=rng_local,
        sequences=sequences,
        logprobs=logprobs,
        cache_u=cache_u,
    )

  final_state = lax.fori_loop(0, seq_len, sampling_iteration, init_state)
  final_logprobs = final_state.logprobs[::beam_size][:, -1].sum(axis=-1)

  # return top beams and corresponding log probs
  return final_state.sequences[::beam_size], final_logprobs
