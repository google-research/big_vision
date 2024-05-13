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

"""gemma reimplementation for big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")

Notes:
- This model only works with pre-trained weights. Weight initialization is not
  properly implemented.
- init seems to be variance(1.0, fan_in, trunc_normal) everywhere.

Example Colab using the models via the PaliGemma decoding logic:
(internal link)
"""


from big_vision.models import common
import big_vision.utils as u
import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import orbax.checkpoint


def get_config(variant):
  """Returns config for specified gemma variant."""
  if variant == "gemma_2b":
    return ml_collections.ConfigDict(
        dict(
            variant=variant,
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            norm_eps=1e-6,
            vocab_size=256_128,
            scan=True,
            remat_policy="nothing_saveable",
        )
    )
  if variant == "gemma_7b":
    return ml_collections.ConfigDict(
        dict(
            variant=variant,
            width=3072,
            depth=28,
            mlp_dim=24_576,
            num_heads=16,
            num_kv_heads=16,
            head_dim=256,
            norm_eps=1e-6,
            vocab_size=256_128,
            scan=True,
            remat_policy="nothing_saveable",
        )
    )
  raise ValueError(f"Unknown variant: {variant}")


def _apply_rope(x, *, positions, max_wavelength=10_000):
  """Applies RoPE positions [B, L] to x [B, L, H, D]."""
  freq_exponents = (2. / x.shape[-1]) * jnp.arange(x.shape[-1] // 2)
  timescale = (max_wavelength ** freq_exponents)
  radians = positions[..., None] / timescale[None, None, :]
  radians = radians[..., None, :]
  # radians.shape = [...,L,1,d=D/2]
  sin, cos = jnp.sin(radians), jnp.cos(radians)
  x1, x2 = jnp.split(x, 2, axis=-1)
  res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return res


def _update_kv_cache(module, k, v, cache_size, cache_dtype):
  """Updates KV cache and returns its current contents."""
  initialized = module.has_variable("cache", "idx")
  batch_size, update_len, num_heads, head_dim = k.shape
  cache_dtype = cache_dtype or k.dtype

  # Idx of which cache row to update next is the same for all examples, so that
  # it allows to update with dynamic_update_slice. But in order to keep things
  # nicely partitioned we store it with leading batch dimension and use only
  # the first entry.
  idx = module.variable("cache", "idx", jnp.zeros, (batch_size,), jnp.int32)

  kv_shape = (batch_size, cache_size, num_heads, head_dim)
  k_cache = module.variable(
      "cache", "k_cache", jnp.zeros, kv_shape, cache_dtype)
  v_cache = module.variable(
      "cache", "v_cache", jnp.zeros, kv_shape, cache_dtype)

  if initialized:  # write k, v in the next cache position.
    assert update_len == 1, update_len
    # Note: idx is the same for all examples. Use value from example 0.
    indices = (0, idx.value[0], 0, 0)
    k_cache.value = jax.lax.dynamic_update_slice(
        k_cache.value, k.astype(cache_dtype), indices)
    v_cache.value = jax.lax.dynamic_update_slice(
        v_cache.value, v.astype(cache_dtype), indices)
    idx.value = idx.value + 1
  else:  # init cache with k, v after padding to cache_size.
    prefill_len = k.shape[1]
    pad_width = ((0, 0), (0, cache_size - prefill_len), (0, 0), (0, 0))
    k_cache.value = jnp.pad(k.astype(cache_dtype), pad_width)
    v_cache.value = jnp.pad(v.astype(cache_dtype), pad_width)
    idx.value = idx.value + prefill_len

  return k_cache.value.astype(k.dtype), v_cache.value.astype(v.dtype)


class Einsum(nn.Module):
  shape: tuple[int, ...]

  @nn.compact
  def __call__(self, eqn, x):
    w = self.param("w", nn.initializers.zeros_init(), self.shape)
    return jnp.einsum(eqn, x, w)


class RMSNorm(nn.Module):

  @nn.compact
  def __call__(self, x):
    scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs


class Embedder(nn.Module):
  """Embedder module."""

  vocab_size: int
  embed_dim: int

  def setup(self):
    self.input_embedding_table = self.param(
        "input_embedding",
        nn.initializers.zeros_init(),
        (self.vocab_size, self.embed_dim),
    )

  def encode(self, x):
    x = self.input_embedding_table[(x,)]
    x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
    return x

  def decode(self, x):
    return jnp.dot(x, self.input_embedding_table.T)


class Attention(nn.Module):
  """Attention module."""

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int

  cache_dtype: str | None = None

  def setup(self):
    if self.num_kv_heads == self.num_heads:
      self.qkv_einsum = Einsum(
          shape=(3, self.num_heads, self.features, self.head_dim),
      )
    else:
      # MQA
      self.q_einsum = Einsum(
          shape=(self.num_heads, self.features, self.head_dim),
      )
      self.kv_einsum = Einsum(
          shape=(2, self.num_kv_heads, self.features, self.head_dim),
      )
    self.attn_vec_einsum = Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

  @nn.compact
  def __call__(self, x, positions, attn_mask, decode, deterministic=True):
    if self.num_kv_heads == self.num_heads:
      q, k, v = self.qkv_einsum("BSD,3KDH->3BSKH", x)
    else:
      q = self.q_einsum("BTD,NDH->BTNH", x)
      k, v = self.kv_einsum("BSD,2KDH->2BSKH", x)

    q = _apply_rope(q, positions=positions)
    q *= self.head_dim**-0.5

    k = _apply_rope(k, positions=positions)
    if decode:
      k, v = _update_kv_cache(self, k, v,
                              cache_size=attn_mask.shape[-1],
                              cache_dtype=self.cache_dtype)

    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.num_kv_heads)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)
    logits = logits.astype(jnp.float32)

    if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
      raise ValueError(
          f"Attention mask with shape {attn_mask.shape} but shapes for q and k "
          f"are: {q.shape} and {k.shape}"
      )

    # big_neg = jnp.finfo(logits.dtype).min
    big_neg = -2.3819763e38  # See gemma/modules.py
    masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

    probs = jax.nn.softmax(masked_logits, axis=-1).astype(k.dtype)

    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
    encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
    attn_output = self.attn_vec_einsum("BTNH,NHD->BTD", encoded)

    return attn_output


class FeedForward(nn.Module):
  """Feed forward module."""

  features: int
  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    w_gating = self.param(
        "gating_einsum",
        nn.initializers.zeros_init(),
        ((2, self.features, self.hidden_dim)),
    )
    ff_gate = jnp.dot(x, w_gating[0])
    gate_value = nn.gelu(ff_gate)

    ff1 = jnp.dot(x, w_gating[1])
    activations = gate_value * ff1

    w_linear = self.param(
        "linear",
        nn.initializers.zeros_init(),
        (self.hidden_dim, self.features),
    )
    outputs = jnp.dot(activations, w_linear)

    return outputs


class Block(nn.Module):
  """Transformer block."""

  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int

  dropout: float = 0.0
  dropout_bdims: tuple[int, ...] = ()
  cache_dtype: str | None = None

  def setup(self):
    self.pre_attention_norm = RMSNorm()
    self.attn = Attention(
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        cache_dtype=self.cache_dtype,
    )
    self.pre_ffw_norm = RMSNorm()
    self.mlp = FeedForward(features=self.embed_dim, hidden_dim=self.hidden_dim)
    if self.dropout:
      self.drop = nn.Dropout(self.dropout, self.dropout_bdims)
    else:
      self.drop = lambda x, _: x

  def __call__(self, x, unused_scan_arg, positions, attn_mask,
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
    return outputs, unused_scan_arg


class Model(nn.Module):
  """gemma model."""

  variant: str

  width: int
  depth: int
  mlp_dim: int
  num_heads: int
  num_kv_heads: int
  head_dim: int
  norm_eps: int
  vocab_size: int

  dropout: float = 0.0
  dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
  cache_dtype: str | None = None

  # TODO: Wire this in all places needed so that the model can be
  # run with different activation dtype. For now only float32 runs.
  embed_dtype: str = "float32"

  scan: bool = False
  remat_policy: str = "none"

  @nn.compact
  def __call__(
      self, tokens, *,
      embedded_prefix=None,
      embed_only=False,
      pre_logits=None,
      positions=None, mask=None,
      decode=False, deterministic=True,
  ):
    """Embed only, or complete forward pass.

    Args:
      tokens: Embedded, then and appended to `embedded_prefix`. Can be None.
      embedded_prefix: Optional prefix that is already embedded.
      embed_only: Whether to compute embeddings only.
      pre_logits: If present computes logits from pre_logits and returns.
      positions: Optional `[B, T]` allows to specify the absolute position of
        the tokens.
      mask: Optional attention mask `[B, T, S]`.
      decode: Whether to use kv-cache. Caller must pass masks and positions.
      deterministic: Forwarded to all dropout layers.

    Returns:
      If `embed_only=False`, then `(logits, out)` will be returned.
      If `embed_only=True`, then the embeddings will be returned.
    """
    out = {}

    embedder = Embedder(
        vocab_size=self.vocab_size,
        embed_dim=self.width,
        name="embedder")

    if pre_logits is not None:
      x = out["pre_logits"] = pre_logits
      logits = out["logits"] = embedder.decode(x)
      return logits, out

    x = []
    if embedded_prefix is not None:
      x.append(embedded_prefix)
    if tokens is not None:
      x.append(embedder.encode(tokens))

    x = jnp.concatenate(x, axis=-2)
    x = x.astype(self.embed_dtype)
    batch_size, seq_len, width = x.shape

    if embed_only:
      return x

    if decode:
      assert positions is not None and mask is not None, (
          "Must explicitly pass positions and mask for decoding.")

    if positions is None:
      positions = jnp.arange(seq_len).astype(jnp.int32)[None, :]
    assert positions.shape[1] == x.shape[1], (positions.shape, x.shape)

    if mask is None:
      mask = nn.attention.make_causal_mask(jnp.ones([batch_size, seq_len]))
    if mask.ndim == 3:
      mask = mask[:, None, :, :]
    cache_size = max(seq_len, mask.shape[-1])
    assert mask.shape == (batch_size, 1, seq_len, cache_size), mask.shape

    if self.remat_policy == "none":
      block_cls = Block
    else:
      block_cls = nn.remat(
          Block,
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
    unused_scan_arg = ()
    for block in blocks:
      x, unused_scan_arg = block(
          x, unused_scan_arg, positions, mask, decode, deterministic)

    assert x.dtype == jnp.dtype(self.embed_dtype)  # Sanity check.
    out["encoded"] = x

    x = RMSNorm(name="final_norm")(x)
    out["pre_logits"] = x

    x = embedder.decode(x)
    out["logits"] = x

    return x, out


_ORBAX_INITS = {}
_BV_INITS = {}


def _load_orbax(path):
  """Loads and coverts Orbax gemma checkpoint."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = checkpointer.restore(path)
  params = flax.traverse_util.unflatten_dict(params, sep="/")["transformer"]
  n = sum(1 for k in params if k.startswith("layer_"))
  params["layers"] = jax.tree.map(
      lambda *xs: np.stack(xs), *(params.pop(f"layer_{i}") for i in range(n))
  )
  mlp = params["layers"]["mlp"]
  mlp["gating_einsum"] = mlp["gating_einsum"].pop("w")
  mlp["linear"] = mlp["linear"].pop("w")
  return params


def _del_pad_rows(params):
  """Some checkpoints have 128 unused padding tokens."""
  emb = params["embedder"]["input_embedding"]
  assert emb.shape[0] == 256_128
  params["embedder"]["input_embedding"] = np.asarray(emb)[:256_000]
  return params


def load(init_params, init_file, model_cfg=None, dont_load=()):
  """Loads existing weights."""
  variant = model_cfg.get("variant", "gemma_2b")
  init_variant = f"{init_file} {variant}"
  if init_variant in _ORBAX_INITS:
    params = _del_pad_rows(_load_orbax(_ORBAX_INITS[init_variant]))
  elif init_variant in _BV_INITS:
    params = _del_pad_rows(u.load_params(_BV_INITS[init_variant]))
  else:
    params = u.load_params(init_file)

  def extend_rows(emb1, target_rows):
    if (missing_rows := target_rows - emb1.shape[0]) == 0:
      return emb1
    assert missing_rows > 0, "You're asking to shrink vocab?!"
    new_rows = np.random.randn(missing_rows, emb1.shape[1])
    new_rows = (new_rows * 0.02).astype(emb1.dtype)
    return np.r_[np.asarray(emb1), new_rows]

  params["embedder"]["input_embedding"] = extend_rows(
      params["embedder"]["input_embedding"],
      model_cfg.vocab_size,
  )

  return common.merge_params(params, init_params, dont_load)
