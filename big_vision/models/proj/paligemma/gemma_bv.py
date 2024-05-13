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

"""Gemma wrapper to make it work for us."""

from big_vision.models.ppp import gemma
import flax.linen as nn
import jax
import jax.numpy as jnp


def _get_config(model):
  config = gemma.get_config(model.variant)
  config.scan = model.scan
  config.remat_policy = model.remat_policy
  if model.vocab_size is not None:
    config.vocab_size = model.vocab_size
  config.dropout = model.dropout
  config.dropout_bdims = model.dropout_bdims
  config.cache_dtype = model.cache_dtype
  return config


@jax.vmap
def _left_to_right_align(x, input_mask, attn_mask):
  """Converts input from left-align to right-aligned."""
  # Due to vmap, this is operating in a single example (not batch level).
  assert x.ndim == 2 and input_mask.ndim == 1 and attn_mask.ndim == 2
  assert x.shape[0] == input_mask.shape[0]
  assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
  seqlen = jnp.sum(input_mask)
  x = jnp.roll(x, -seqlen, axis=0)
  input_mask = jnp.roll(input_mask, -seqlen, axis=0)
  attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
  return x, input_mask, attn_mask


class Model(nn.Module):
  """Wrapping gemma big_vision model."""
  variant: str = "gemma_2b"
  scan: bool = True
  remat_policy: str = "nothing_saveable"
  vocab_size: int | None = None

  dropout: float = 0.0
  dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
  cache_dtype: str | None = "bfloat16"  # bfloat16 to save memory and transfers.

  def setup(self):
    # The parent+name avoids an unnecessary nesting in params pytree.
    self.model = gemma.Model(**_get_config(self), parent=self.scope, name="")

  def embed_tokens(self, tokens, train=False):
    # Turns int32[B,T] tokens into float32[B,T,d_model] embeddings.
    # Really just the vocab embedding.
    return self.model(tokens, embed_only=True, deterministic=not train)

  def compute_logits(self, pre_logits, train=False):
    return self.model(None, pre_logits=pre_logits, deterministic=not train)[0]

  def __call__(self, embs, mask=None, train=False):
    # Turns float32[B,T,d_model] embedding sequence to logits.
    # call(emb_tokens(tokens)) should be a forward pass.
    # Allow for specifying int32[B,T,T] attention masks. For convenience
    # default to triangular autorgressive mask when None, but not P0.
    # Return float32[B,T,vocab_size] logits and out-dict.

    batch_size, _, d_model = embs.shape
    assert d_model == self.embdim
    logits, out = self.model(
        tokens=jnp.zeros([batch_size, 0], dtype=jnp.int32),
        embedded_prefix=embs,
        mask=mask,
        deterministic=not train,
    )
    return logits, out

  def prefill_cache(self, x, input_mask, attn_mask, *, cache_size):
    """Initializes decoding cache with `x` [B, N, E] as prompt.

    IMPORTANT: Inputs MUST be left-aligned and attn_mask should not allow
    input tokens to attend to padding tokens.

    TODO: Relax left-align requirement by converting any input into
    a right aligned input with no attention to padding tokens.

    Args:
      x: float[B, N, E] with prompt tokens.
      input_mask: bool[B, N]. True indicates tokens are part of the prompt.
        False indicates padding tokens. This class doesn't combine this with
        attn_mask, so mask out the attention to padding tokens beforehand.
      attn_mask: bool[B, N, N]. Indicates which tokens can attend to which while
        processing the prompt tokens. During extend_cache tokens, it is assumed
        that tokens can attend all previous valid tokens.
      cache_size: int. Indicates the size of the cache. The prompt will consume
        the first N entries of the cache. Each subsequent extend_cache will
        consume one entry. Behaviour is undefined when prefill_len plus number
        of extend_cache exceeds the cache_size.

    Returns:
      logits of the last valid token (i.e. last logits where input_mask=True).
    """
    # To call the model with decode=True we need to be able to provide:
    #   (a) positions of tokens [B, N], ([B, 1] for extend)
    #   (b) attention mask [B, N, cache_size] ([B, 1, cache_size] for extend)
    #
    # To do so we track how many tokens each example has seen so far, and we
    # align the prompt to the right so that cache usage for each example is in
    # a continuous subsequent of (cache_begin, cache_end] such that cache_end
    # is the same for all sequences (this allows to do faster row updates of
    # the cache during decoding).
    x, input_mask, attn_mask = _left_to_right_align(x, input_mask, attn_mask)

    # Track sequence len
    seq_len = jnp.sum(input_mask, axis=-1)
    self.put_variable("cache", "seq_len", seq_len)
    positions = jnp.cumsum(input_mask, axis=-1) - 1

    # Initialize cache_begin and cache_end. Note: cache_end is the same for all
    # sequences but we keep it per example to allow easy sharding rules with
    # batch as the first axis.
    batch_size, prefill_len, _ = x.shape
    self.put_variable("cache", "cache_begin", prefill_len - seq_len)
    self.put_variable(
        "cache", "cache_end", jnp.full((batch_size,), prefill_len, jnp.int32)
    )

    # Pad attention to set the cache size.
    mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, cache_size - prefill_len)))

    _, aux = self.model(
        tokens=None,
        embedded_prefix=x,
        positions=positions,
        mask=mask,
        decode=True,
    )
    return self.compute_logits(aux["pre_logits"][:, -1:])

  def extend_cache(self, x):
    """Extends decoding cache with `x` [B, 1, E] and returns logits."""
    assert x.shape[1] == 1, "Only supports extend the cache by one token."
    if self.model.scan:
      cache_size = self.variables["cache"]["layers"]["attn"]["k_cache"].shape[2]
    else:
      raise NotImplementedError("Not implemented yet.")

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

    logits, _ = self.model(
        tokens=None, embedded_prefix=x,
        positions=positions[:, None], mask=mask, decode=True)
    return logits

  @property
  def embdim(self):
    return _get_config(self).width


load = gemma.load
