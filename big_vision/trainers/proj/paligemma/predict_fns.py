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

"""Prediction functions for PaliGemma."""

import collections
import functools

from big_vision.pp import registry
import big_vision.utils as u
import einops
import jax
import jax.numpy as jnp
import numpy as np


P = jax.sharding.PartitionSpec

# pylint: disable=missing-function-docstring


def get_all(model):
  """Returns `predict_fns` for evaluators."""
  fns = {
      "logits": _logits,
      "image_avg_repr": _image_avg_repr,
      "decode": _decode,
      "decode_with_logp": _decode_with_logp,
      "beam_decode": _beam_decode,
  }
  return {name: functools.partial(fn, model=model) for name, fn in fns.items()}


def _logits(train_state, batch, *, model):
  images, text, mask = batch["image"], batch["text"], batch["mask_ar"]
  text_logits, out = model.apply(
      {"params": train_state["params"]},
      images, text[:, :-1], mask[:, :-1],
  )
  return text_logits, out


def _image_avg_repr(train_state, batch, *, model, key="img/pre_logits"):
  zimg, out = model.apply(
      {"params": train_state["params"]},
      image=batch["image"],
      method=model.embed_image,
  )
  if key:
    zimg = u.tree_get(out, key)
  # At this point, zimg is a (batch of) sequence of image tokens, because we
  # assume the model is a vit with "none" head. This predict-fn is for fewshot
  # evaluator, so we need to turn it into reasonably-sized vector -> avg.
  zimg = jnp.mean(zimg, axis=range(1, zimg.ndim - 1))
  return zimg, out


def _decode_with_logp(
    train_state, batch, *, model, devices, max_decode_len, eos_token,
    best_of_n=1, sampler="greedy", eos_look_behind=0):
  """Sample token continuations to the input sequences."""
  mesh = jax.sharding.Mesh(devices, ("devices",))
  replicate_sharding = jax.sharding.NamedSharding(mesh, P())
  bs_shardable = len(batch["image"]) % jax.device_count() == 0
  out_sharding = jax.sharding.NamedSharding(
      mesh, P("devices") if bs_shardable else P()
  )

  # Prefill the model cache and generate logits for first token.
  logits, cache = jax.jit(
      _prefill_cache,
      out_shardings=(None, out_sharding),
      static_argnames=("model", "max_decode_len"),
  )(
      train_state["params"],
      {
          "image": batch["image"],
          "text": batch["text"],
          "mask_input": batch["mask_input"],
          "mask_ar": batch["mask_ar"],
      },
      model=model,
      max_decode_len=max_decode_len,
  )

  # Mask indicating real examples. False if example is used to pad the batch.
  mask = batch["_mask"]

  # Mask indicating tokens for which the logits will be set to -Inf. Can be a
  # Boolean mask or indices.
  tok_mask = batch.get("mask_logits", None)

  # Repeat example in case we are picking the best of n.
  logits, cache, mask = jax.jit(
      _bon_repeat,
      static_argnames=("n",)
  )((logits, cache, mask), n=best_of_n)

  decode_sample_output = jax.jit(
      _decode_sample_output,
      static_argnames=("max_decode_len", "sampler"),
  )
  decode_early_stop = jax.jit(
      _decode_early_stop,
      out_shardings=replicate_sharding,
      static_argnames=("eos_token",),
  )
  extend_cache = jax.jit(
      _extend_cache,
      donate_argnums=1,
      out_shardings=(None, out_sharding),
      static_argnames=("model",),
  )

  # Keep sampling tokens from last logits until EOS or max_decode_len.
  state = None
  # Setting `eos_look_behind>0` removes blocking transfer with small batches.
  stops = collections.deque(maxlen=1 + eos_look_behind)
  for idx in range(max_decode_len):
    tokens, state = decode_sample_output(
        state, logits, tok_mask, max_decode_len=max_decode_len, sampler=sampler
    )

    if idx + 1 >= max_decode_len:
      break

    stops.append(decode_early_stop(state, mask, eos_token=eos_token))
    if len(stops) == stops.maxlen and jax.device_get(stops[0]):
      break

    # Compute logits for next token
    logits, cache = extend_cache(
        train_state["params"], cache, tokens, model=model
    )

  # Select the best of n sample for each example.
  _, tokens, logp = jax.jit(
      _bon_select,
      out_shardings=out_sharding,
      static_argnames=("n", "eos_token"),
  )(state, n=best_of_n, eos_token=eos_token)

  return tokens, logp


def _decode(train_state, batch, **kwargs):
  tokens, _ = _decode_with_logp(train_state, batch, **kwargs)
  return tokens


def _bon_repeat(tree, *, n):
  return jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), tree)


def _compute_score(tokens, logp, eos_token):
  """Compute log-probability of each sequence up to first eos (including it)."""
  seqlen = jnp.sum(jnp.cumsum(tokens == eos_token, axis=-1) == 0, axis=-1) + 1
  token_mask = jnp.arange(tokens.shape[-1]) < seqlen[..., None]
  scores = jnp.sum(logp * token_mask, axis=-1)
  return scores


def _bon_select(state, *, n, eos_token):
  """Pick the sampled sequence with the highest likelihood for each example."""
  (_, tokens, logp) = state

  # Filter state to only keep the best of each example.
  scores = _compute_score(tokens, logp, eos_token)
  scores = einops.rearrange(scores, "(b n) -> b n", n=n)
  state = jax.tree.map(
      lambda x: einops.rearrange(x, "(b n) l -> b n l", n=n), state)
  best_indices = jnp.argmax(scores, -1)  # [b]
  state = jax.tree.map(
      lambda x: jnp.take_along_axis(x, best_indices[:, None, None], axis=1),
      state)
  state = jax.tree.map(lambda x: x[:, 0], state)

  return state


def _decode_sample_output(state, logits, tok_mask, *, max_decode_len, sampler):
  if state is None:
    # Decode state keeps track of sampled tokens and their logp.
    bs = logits.shape[0]
    seqlen = jnp.zeros((bs, 1), dtype=jnp.int32)
    tokens = jnp.zeros((bs, max_decode_len), dtype=jnp.int32)
    logp = jnp.zeros((bs, max_decode_len), dtype=logits.dtype)
  else:
    (seqlen, tokens, logp) = state

  # Sample tokens.
  sampled_tokens, sampled_logp = _sample_logits(logits, sampler=sampler,
                                                tok_mask=tok_mask)

  # Update state with sampled outputs.
  new_len = seqlen + 1
  new_tokens = _put_along_last_axis(tokens, seqlen, sampled_tokens)
  new_logp = _put_along_last_axis(logp, seqlen, sampled_logp)
  new_state = (new_len, new_tokens, new_logp)

  return sampled_tokens, new_state


def _decode_early_stop(state, mask, *, eos_token):
  (seqlen, tokens, unused_logp) = state
  token_mask = jnp.arange(tokens.shape[-1])[None, :] < seqlen
  has_eos = jnp.any(jnp.logical_and(tokens == eos_token, token_mask), axis=-1)
  done = jnp.logical_or(has_eos, jnp.logical_not(mask))
  return jnp.all(done)


def _put_along_last_axis(arr, indices, values):
  """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
  assert arr.ndim == indices.ndim == values.ndim, (
      arr.ndim, indices.ndim, values.ndim)
  onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
  put_mask = jnp.einsum("...i,...in->...n",
                        jnp.ones(values.shape, jnp.int32), onehot)
  put_values = jnp.einsum("...i,...in->...n", values, onehot)
  return jnp.where(put_mask, put_values, arr)


def _prefill_cache(params, batch, *, model, max_decode_len):
  """Initialize the model cache for decoding with the prompts."""
  variables = {"params": params}
  (x, input_mask, mask_ar), _ = model.apply(
      variables, batch["image"], batch["text"],
      input_mask=batch["mask_input"],
      mask_ar=batch["mask_ar"],
      method=model.embed_image_and_text)
  last_logits, variables = model.apply(
      variables, x, input_mask, mask_ar,
      cache_size=x.shape[1] + max_decode_len,
      method=model.prefill_cache,
      mutable=("cache",))
  return last_logits, variables["cache"]


def _extend_cache(params, cache, tokens, *, model):
  """Extend the model cache for decoding with one token per sequence."""
  variables = {"params": params, "cache": cache}
  x, _ = model.apply(variables, tokens, method=model.embed_text)
  last_logits, variables = model.apply(
      variables, x, method=model.extend_cache, mutable=("cache",))
  return last_logits, variables["cache"]


def _sample_logits(logits, sampler, tok_mask=None):
  """Returns a sampled token and its logp from logits."""
  # Note: Consider making it possible for evaluators to pass rng seed to
  # decode functions. For now generate it from jax.lax and avoid evaluators
  # having to deal with it.
  rng = jax.random.PRNGKey(
      jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))

  masked_logits = logits
  if tok_mask is not None:
    masked_logits = masked_logits.at[..., tok_mask].set(-jnp.inf)

  # Use Registry to support specifying things like:
  #  "greedy", "nucleus(0.2)", "temperature(t=1.0)"
  sampled_tokens = registry.Registry.lookup("paligemma_sampler." + sampler)(
      logits=masked_logits, rng=rng)

  # Find the log probability (normalized logits) of selected tokens.
  # NOTE: If you use tok_mask this returns the probability of the tokens while
  # ignoring the masking. This is useful for pix2seq-style which has tokens like
  # "noise" which it does not want to sample but it wants to use it to affect
  # the score/logp of classes being sampled and it wants to intrepet as a
  # confidence.
  sampled_logp = jnp.take_along_axis(
      jax.nn.log_softmax(logits, axis=-1),
      sampled_tokens[..., None], -1)[..., 0]

  return sampled_tokens, sampled_logp


@registry.Registry.register("paligemma_sampler.greedy")
def _greedy_sampling(*, logits, rng):
  del rng
  return jnp.argmax(logits, axis=-1)


@registry.Registry.register("paligemma_sampler.temperature")
def _temperature_sampling(t, *, logits, rng):
  return jax.random.categorical(rng, logits / t)


@registry.Registry.register("paligemma_sampler.nucleus")
def _nucleus_sampling(p: float, t: float = 1.0, *, logits, rng):
  logits = logits / t
  neg_inf = np.array(-1.0e7)  # Effective negative infinity.
  logits_sorted = jnp.sort(logits, axis=-1, descending=True)
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
  cutoff_index = jnp.sum(sorted_cum_probs < p, axis=-1, keepdims=True)
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit,
                     jnp.full_like(logits, neg_inf), logits)
  return jax.random.categorical(rng, logits)


def _beam_decode(train_state, batch, *,
                 model, devices, max_decode_len,
                 eos_token, beam_size):
  """Beam search (greedy/top-k exploration)."""
  mesh = jax.sharding.Mesh(devices, ("devices",))
  replicate_sharding = jax.sharding.NamedSharding(mesh, P())
  bs_shardable = len(batch["image"]) % jax.device_count() == 0
  out_sharding = jax.sharding.NamedSharding(
      mesh, P("devices") if bs_shardable else P()
  )

  # Prefill the model cache and generate logits for first token.
  logits, cache = jax.jit(
      _prefill_cache,
      out_shardings=(None, out_sharding),
      static_argnames=("model", "max_decode_len"),
  )(
      train_state["params"],
      {
          "image": batch["image"],
          "text": batch["text"],
          "mask_input": batch["mask_input"],
          "mask_ar": batch["mask_ar"],
      },
      model=model,
      max_decode_len=max_decode_len,
  )

  # Mask indicating real examples. False if example is used to pad the batch.
  mask = batch["_mask"]

  beam_sample_output = jax.jit(
      _beam_sample_output,
      donate_argnums=2,
      out_shardings=(None, None, out_sharding),
      static_argnames=("max_decode_len", "beam_size", "eos_token"),
  )
  beam_early_stop = jax.jit(
      _beam_early_stop,
      out_shardings=replicate_sharding,
      static_argnames=("eos_token",),
  )
  extend_cache = jax.jit(
      _extend_cache,
      donate_argnums=1,
      out_shardings=(None, out_sharding),
      static_argnames=("model",),
  )

  # Keep sampling tokens from last logits until EOS or max_decode_len.
  state = None
  for idx in range(max_decode_len):
    tokens, state, cache = beam_sample_output(
        state, logits, cache,
        max_decode_len=max_decode_len, beam_size=beam_size, eos_token=eos_token)

    early_stop = beam_early_stop(state, mask, eos_token=eos_token)
    if jax.device_get(early_stop) or (idx + 1 >= max_decode_len):
      break

    # Compute logits for next token
    logits, cache = extend_cache(
        train_state["params"], cache, tokens, model=model)

  return jax.jit(_beam_make_output, out_shardings=out_sharding)(state)


def _beam_early_stop(state, mask, eos_token):
  (best_tokens, best_logp, seqlen, unused_tokens, logp) = state

  # Scores of finalized sequences.
  best_scores = _compute_score(best_tokens, best_logp, eos_token)

  # Scores of live sequences.
  live_mask = jnp.arange(logp.shape[-1])[None, None] < seqlen
  live_scores = jnp.sum(logp * live_mask, axis=-1)
  live_scores = jnp.max(live_scores, axis=1)

  done = live_scores < best_scores
  return jnp.all(jnp.logical_or(done, jnp.logical_not(mask)))


def _beam_make_output(state):
  (best_tokens, *_) = state
  return best_tokens[:, 0, ...]


def _beam_sample_output(state, logits, cache, *,
                        beam_size, max_decode_len, eos_token):
  assert logits.shape[1] == 1
  logits = jax.nn.log_softmax(logits[:, 0, :])  # Normalize logits

  if state is None:
    bs = logits.shape[0]
    # Beam decode state keeps track of:
    # A) Best sampled output for each example. At initialization these have
    # shape[1]=0, but end up with shape[1]=1 after first call.
    best_tokens = jnp.zeros((bs, 0, max_decode_len), dtype=jnp.int32)
    best_logp = jnp.zeros((bs, 0, max_decode_len), dtype=logits.dtype)
    # B) N candidate sequences for each example. At initialization these have
    # beam_size=1, but end up with correct beam_size when expanded.
    seqlen = jnp.zeros((bs, 1, 1), dtype=jnp.int32)
    tokens = jnp.zeros((bs, 1, max_decode_len), dtype=jnp.int32)
    logp = jnp.zeros((bs, 1, max_decode_len), dtype=logits.dtype)
  else:
    (best_tokens, best_logp, seqlen, tokens, logp) = state
    bs = logits.shape[0] // beam_size
    assert best_tokens.shape[0] == bs

  # Reshape cache to [example, candidate, ...].
  # Note: on first call the number of candidates is 1. Later it is beam_size.
  cache, logits = jax.tree.map(
      lambda x: einops.rearrange(x, "(b n) ... -> b n ...", b=bs),
      (cache, logits))

  # Consider a live sequence could end now and update the best finished
  # sequences so far for each example. This strategy is found in some beam
  # implementations such as in praxis.
  # The code below also adjusts the best shape[1]=0 -> 1 during first call.
  eos_tokens = jnp.array(eos_token)[None, None, None]
  new_tokens = _put_along_last_axis(tokens, seqlen, eos_tokens)
  new_logp = _put_along_last_axis(logp, seqlen, logits[:, :, eos_token, None])

  best_tokens = jnp.concatenate([best_tokens, new_tokens], axis=1)
  best_logp = jnp.concatenate([best_logp, new_logp], axis=1)
  best_scores = _compute_score(best_tokens, best_logp, eos_token=eos_token)
  _, top_indices = jax.lax.top_k(best_scores, k=1)

  best_tokens = jnp.take_along_axis(best_tokens, top_indices[..., None], axis=1)
  best_logp = jnp.take_along_axis(best_logp, top_indices[..., None], axis=1)

  # To find the next best N live candidates we expand each candidate and keep
  # the best N (ignoring EOS tokens). In this case we expand into (N+1)
  # candidates and set their likelihood to "-inf" (if EOS) after the fact.
  live_mask = jnp.arange(logp.shape[-1])[None, None] < seqlen
  live_scores = jnp.sum(logp * live_mask, axis=-1)
  topk_logits, topk_tokens = jax.lax.top_k(logits, beam_size+1)
  scores = live_scores[..., None] + topk_logits
  scores = jnp.where(
      topk_tokens != eos_token, scores, jnp.finfo(scores.dtype).min)

  # From the N*(N+1) candidates find the top N for each example.
  topk_logits, topk_tokens, scores = jax.tree.map(
      lambda x: einops.rearrange(x, "b n np1 -> b (n np1)"),
      (topk_logits, topk_tokens, scores))
  _, topk_indices = jax.lax.top_k(scores, k=beam_size)
  sampled_indices = topk_indices // (beam_size+1)
  sampled_tokens = jnp.take_along_axis(
      topk_tokens, topk_indices, axis=-1)[..., None]
  sampled_logits = jnp.take_along_axis(
      topk_logits, topk_indices, axis=-1)[..., None]

  # Adjust cache and state so it matches the selected top N input candidates.
  # This also adjusts the beam_size=1->n during first call.
  def take_candidates(x):
    one_hot_matrix = jax.nn.one_hot(sampled_indices, x.shape[1], dtype=x.dtype)
    return jnp.einsum("bi...,boi->bo...", x, one_hot_matrix)
  cache, seqlen, tokens, logp = jax.tree.map(
      take_candidates, (cache, seqlen, tokens, logp))

  # Write the sampled tokens/logits on the reshuffled state.
  tokens = _put_along_last_axis(tokens, seqlen, sampled_tokens)
  logp = _put_along_last_axis(logp, seqlen, sampled_logits)
  seqlen = seqlen + 1

  state = (best_tokens, best_logp, seqlen, tokens, logp)

  # Reshape to [(example, candidate), ...].
  sampled_tokens, cache = jax.tree.map(
      lambda x: einops.rearrange(x, "b n ... -> (b n) ..."),
      (sampled_tokens, cache))

  return sampled_tokens, state, cache
