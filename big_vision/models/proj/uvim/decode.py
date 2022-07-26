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

"""Inference."""
import functools

from typing import Any, Callable, Optional, Tuple

import flax
from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp

import numpy as np


EOS_ID = 1
NEG_INF = np.array(-1.0e7)  # Effective negative infinity.


GenerateFn = Callable[...,
                      Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]]


def temperature_sampling(*args, temperature=1.0, top_k=0, top_p=0.0, **kwargs):
  """Convenience wrapper for temperature sampling."""
  return generate(*args, generate_fn=_temperature_sampling,
                  temperature=temperature,
                  top_k=top_k,
                  top_p=top_p,
                  **kwargs)


def topk_sampling(*args, temperature=1.0, top_k=20, **kwargs):
  """Convenience wrapper for top-k sampling."""
  return generate(*args, generate_fn=_temperature_sampling,
                  temperature=temperature,
                  top_k=top_k,
                  top_p=0.0,
                  **kwargs)


def nucleus_sampling(*args, temperature=1.0, top_p=0.2, **kwargs):
  """Convenience wrapper for nucleus sampling."""
  return generate(*args, generate_fn=_temperature_sampling,
                  temperature=temperature,
                  top_k=0,
                  top_p=top_p,
                  **kwargs)


def argmax_sampling(*args, **kwargs):
  """Convenience wrapper for argmax sampling."""
  return generate(*args, generate_fn=_temperature_sampling,
                  temperature=1e-7,
                  top_k=0,
                  top_p=0.0,
                  **kwargs)


def generate(params, inputs, prompts, seed, *,
             model: nn.Module,
             generate_fn: GenerateFn,
             num_samples: int = 1,
             prefill: bool = False,
             eos_token: int = EOS_ID,
             **generate_fn_kwargs):
  """Generate sequence with fast decoding beam search on a batch.

  Model must support:
    encode(inputs) -> encoded, or encode(*inputs) -> encoded.
    decode(encoded, prompts, decode=True/False, max_decode_length) -> logits

  Args:
    params: model parameters.
    inputs: either a single `jnp.ndarray` of e.g. images, or
      a tuple of inputs which are passed via `model.encode(*inputs)`.
    prompts: [batch_size, max_decode_len] forced tokens for generation.
      prompts need to finish with 0 token, they should not contain the end
      markers. If no prompting is required, pass an all zeros tensor.
    seed: PRNG key for random sampling.
    model: object with methods encode and decode.
    generate_fn: search or sampling function to generate sequences.
    num_samples: number of samples to generate per item.
    prefill: whether to prefill cache.
    eos_token: if of end-of-sentence token for target vocabulary.
    **generate_fn_kwargs: generate fn specific kwargs.

  Returns:
    Top-scoring sequences (worst scores first).
      [batch_size, num_samples, max_decode_len]
    Scores of the generated sequences (worst scores first). The
      returned scores are modified log probabilities. May be absent.
      [batch_size, max_decode_len]
    Log probs for the generated tokens. May be absent.
      [batch_size, num_samples, max_decode_len]
  """
  _, max_decode_len = prompts.shape
  decode_kwargs = {"max_decode_length": max_decode_len}

  def encode(model, inputs):
    if not isinstance(inputs, tuple):
      inputs = (inputs,)
    return model.encode(*inputs)

  encoded_inputs = nn.apply(encode, model)(params, inputs)
  if isinstance(encoded_inputs, tuple):
    encoded_inputs, enc_pos_emb = encoded_inputs
    decode_kwargs["enc_pos_emb"] = enc_pos_emb

  def init_cache(model):
    encoded = jnp.zeros_like(encoded_inputs)
    targets = jnp.zeros_like(prompts)
    return model.decode(encoded, targets, decode=True, **decode_kwargs)

  cache = nn.apply(init_cache, model, mutable=True)(params)[1]["cache"]

  def prefill_cache(model, encoded, targets):
    return model.decode(encoded, targets, prefill=True, **decode_kwargs)

  if prefill:
    cache = nn.apply(prefill_cache, model, mutable=True)(
        {"params": params["params"], "cache": cache},
        encoded_inputs, prompts)[1]["cache"]

  def tokens_to_logits(tokens, cache):
    def decode_step(model, tokens):
      encoded = expand_samples_dim_and_flatten(
          encoded_inputs, num_samples)
      return model.decode(encoded, tokens, decode=True, **decode_kwargs)

    logits, aux = nn.apply(decode_step, model, mutable=True)(
        {"params": params["params"], "cache": cache}, tokens)
    return logits.squeeze(axis=1), aux["cache"]

  beam_seqs, scores, logprobs = generate_fn(
      prompts,
      cache,
      tokens_to_logits,
      num_samples=num_samples,
      eos_token=eos_token,
      max_decode_len=max_decode_len,
      seed=seed,
      **generate_fn_kwargs)
  return beam_seqs, scores, logprobs


def expand_samples_dim(x, num_samples):
  """Creates new dimension in non-scalar array and tiles into it."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  x = jnp.expand_dims(x, axis=1)
  tile_dims = [1] * x.ndim
  tile_dims[1] = num_samples
  return jnp.tile(x, tile_dims)


def flatten_samples_dim(x):
  """Flattens samples dim into batch dim."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unflatten_samples_dim(x, batch_size, num_samples):
  """Unflattens first dim into batch and samples dims."""
  if x.ndim == 0:  # ignore scalars (e.g. cache index)
    return x
  assert batch_size * num_samples == x.shape[0]
  return x.reshape((batch_size, num_samples) + x.shape[1:])


def expand_samples_dim_and_flatten(x, num_samples):
  """Expands the each batch item by num_samples in batch dimension."""
  return flatten_samples_dim(expand_samples_dim(x, num_samples))


def cache_map(fn, cache):
  """Maps function over caches, even multiple caches in various layers."""
  frozen = isinstance(cache, flax.core.FrozenDict)
  if frozen:
    cache = flax.core.unfreeze(cache)
  flat_cache = flax.traverse_util.flatten_dict(cache)
  # Exclude cached relative position bias from beam expansion, etc.
  keyvals = {k: v for k, v in flat_cache.items() if k[-1] != "cached_bias"}
  keyvals = jax.tree_map(fn, keyvals)
  flat_cache.update(keyvals)
  new_cache = flax.traverse_util.unflatten_dict(flat_cache)
  if frozen:
    new_cache = flax.core.freeze(new_cache)
  return new_cache


@flax.struct.dataclass
class LoopState:
  """Internal state of the temperature sampling loop."""
  # Position in the sequence that we are currently looking at.
  cur_index: int
  # Cache for fast auto-regressive decoding.
  cache: Any
  # Flags indicating whether the sequence reached eos [B*N].
  flags_finished: jnp.ndarray
  # Sequences being generated [B*N, L+1]. Note: sequences start with 0 token.
  sequences: jnp.ndarray
  scores: jnp.array  # Total sequence scores per batch element [B*N].
  logprobs: jnp.array  # Logprobs of selected tokens [B*N, L].
  rng: jnp.ndarray  # PRNGKey of the loop state.


def _init_state(prompts, cache, init_rng_key, num_samples):
  batch_size, max_decode_len_plus_one = prompts.shape
  # Add extra samples dim to attention cache pytree elements.
  cache = cache_map(
      lambda x: expand_samples_dim_and_flatten(x, num_samples), cache)
  return LoopState(
      cur_index=0,
      cache=cache,
      flags_finished=jnp.zeros((batch_size*num_samples), dtype=jnp.bool_),
      sequences=expand_samples_dim_and_flatten(prompts, num_samples),
      scores=jnp.zeros((batch_size*num_samples)),
      logprobs=jnp.zeros((batch_size*num_samples, max_decode_len_plus_one-1)),
      rng=init_rng_key)


def _should_temperature_sampling_continue(state, max_decode_len):
  """Check if we should continue or not."""

  max_length_not_reached = state.cur_index < max_decode_len - 1
  all_seqs_finished = jnp.all(state.flags_finished)
  return max_length_not_reached & (~all_seqs_finished)


def _temperature_sampling_iteration(state, tokens_to_logits, temperature, eos,
                                    top_k, top_p, mask_token_ids=()):
  """Temperature sampling step function."""

  rng_sampling, rng = jax.random.split(state.rng)

  # 1. Use the model to generate a distribution over the vocabulary (for the
  # next token) and sample from it, optionally applying the temperature.
  # --> [B,].
  cur_tokens = state.sequences[:, state.cur_index]
  logits, new_cache = tokens_to_logits(cur_tokens[:, None], state.cache)
  assert logits.ndim == 2, ("tokens_to_logits expected to return a"
                            f"2-dimensional array [B, V], got {logits.ndim}"
                            "dimensions.")
  logprobs = jax.nn.log_softmax(logits)

  # Do not sample special tokens in with ids in mask_token_ids.
  if mask_token_ids:
    probs = jax.nn.softmax(logits)
    for i in mask_token_ids:
      probs = probs.at[:, i].set(0.)
    probs = probs / jnp.sum(probs, -1, keepdims=True)
    logits = jnp.log(probs)

  if top_p:  # Nucleus sampling.
    logits_sorted = jnp.sort(logits, axis=-1)[:, ::-1]
    sorted_cum_probs = jnp.cumsum(
        jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
    cutoff_index = jnp.sum(sorted_cum_probs < top_p, axis=-1, keepdims=True)
    cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
    logits = jnp.where(logits < cutoff_logit,
                       jnp.full_like(logits, NEG_INF), logits)
  if top_k:
    topk_logits, topk_indices = jax.lax.top_k(logits, top_k)
    topk_token = jax.random.categorical(rng_sampling, topk_logits / temperature)
    sampled_tokens = jnp.squeeze(
        jnp.take_along_axis(topk_indices, jnp.expand_dims(topk_token, -1),
                            axis=-1), axis=-1)
  else:
    sampled_tokens = jax.random.categorical(rng_sampling, logits / temperature)

  sampled_logprobs = jnp.squeeze(jnp.take_along_axis(
      logprobs, jnp.expand_dims(sampled_tokens, axis=1), axis=-1), axis=-1)

  # 2. Use the sampled tokens to update the sequences that did not finish yet,
  # but only if they are out of prompt.
  next_tokens = state.sequences[:, state.cur_index + 1]
  next_logprobs = jnp.squeeze(jnp.take_along_axis(
      logprobs, jnp.expand_dims(next_tokens, axis=1), axis=-1), axis=-1)
  out_of_prompt = next_tokens == 0
  update_pos = out_of_prompt * (~state.flags_finished)
  next_tokens = sampled_tokens * update_pos + next_tokens * (~update_pos)
  sampled_logprobs = update_pos*sampled_logprobs + ~update_pos*next_logprobs
  sequences = state.sequences.at[:, state.cur_index + 1].set(next_tokens)
  scores = state.scores + sampled_logprobs
  seqs_logprobs = state.logprobs.at[:, state.cur_index].set(sampled_logprobs)

  # 3. Update the finished flags. Only out of prompts seqs can finish.
  flags_finished = out_of_prompt & (state.flags_finished |
                                    (sampled_tokens == eos))
  return LoopState(
      cur_index=state.cur_index+1,
      cache=new_cache,
      flags_finished=flags_finished,
      sequences=sequences,
      scores=scores,
      logprobs=seqs_logprobs,
      rng=rng)


def _temperature_sampling(prompts, cache, tokens_to_logits, num_samples=1,
                          eos_token=EOS_ID, max_decode_len=None,
                          seed=0, temperature=1., top_k=0, top_p=0.0,
                          mask_token_ids=()):
  """Temperature sampling.

  Purely stochastic sampling-based greedy procedure to generate sequences. Every
  next token in the sequence is sampled from the discrete vocab distribution
  produced by the auto-regressive sequence model. Optionally we can adjust the
  distribution by changing the temperature before sampling from it. Generated
  sequences are no longer than max_decode_len.

  Args:
    prompts: optional prompts [B, L]. By default (None), we call free form
      generation without any prompts. Prompt sequences should finish with
      trailing zeros and should not contain eos tokens.
    cache: cache for fast decoding (generation).
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    num_samples: int: number of samples to generate per batch item. Note, no
      deduplication is performed, and in dependence of parameter settings, same
      sequences could be generated and returned.
    eos_token: end-of-sentence token.
    max_decode_len: maximal length of generated sequences (L).
    seed: PRNGKey for random sampling.
    temperature: positive real-valued sampling temperature. By default we sample
      from the original distribution. As the temperature approaches 0., the
      entire distribution concentrates on the most probable outcome(s).
    top_k: limit sampling to only top-k logits. Zero means no limit.
    top_p: limit sampling to smallest number of top logits with max cumulative
      prob <= top_p. Zero means no limit. Cannot use both top_p and top_k.
    mask_token_ids: if set then tokens with given ids are not sampled.

  Returns:
    sequences: generated sequences [B, num_samples, L].
    scores: not implemented in the naive temperature sampling [B, num_samples].
    logprobs: Log probabilities for the generated tokens [B, num_samples, L].
  """
  if top_k > 0 and top_p > 0.0:
    raise ValueError(f"Cannot use both top_k {top_k} and top_p {top_p}.")
  if max_decode_len is None:
    max_decode_len = prompts.shape[1]
  # We will start generating sequences from 0 token.
  prompts = jnp.pad(prompts, ((0, 0), (1, 0)))
  eos = jnp.array(eos_token)
  if isinstance(seed, int):
    seed = jax.random.PRNGKey(seed)

  # Initialize the state.
  loop_init_state = _init_state(prompts, cache, seed, num_samples)
  should_temperature_sampling_continue_fn = functools.partial(
      _should_temperature_sampling_continue,
      max_decode_len=max_decode_len+1)  # Account for prompt padding with 0's.
  temperature_sampling_iteration_fn = functools.partial(
      _temperature_sampling_iteration,
      tokens_to_logits=tokens_to_logits,
      temperature=temperature, top_k=top_k, top_p=top_p,
      eos=eos, mask_token_ids=mask_token_ids)

  # Run the temperature sampling and generate the sequences.
  final_state = lax.while_loop(
      should_temperature_sampling_continue_fn,
      temperature_sampling_iteration_fn,
      loop_init_state)

  # Return the generated sequences, discarding the 0 token in the beginning.
  return (
      final_state.sequences[:, 1:].reshape((-1, num_samples, max_decode_len)),
      final_state.scores.reshape((-1, num_samples)),
      final_state.logprobs.reshape((-1, num_samples, max_decode_len)))
