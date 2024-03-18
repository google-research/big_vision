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

"""Utilities shared across models."""

from absl import logging
import big_vision.utils as u
import flax.linen as nn
import jax
import jax.numpy as jnp


def merge_params(loaded, inited, dont_load=(), match_dtype=False):
  """Makes `loaded` pytree match `init`, warning or failing on mismatch.

  Args:
    loaded: pytree of parameters, typically loaded from a checkpoint.
    inited: pytree of parameter, typically coming from model init.
    dont_load: List of regexes for parameters which shall not be taken
      from `loaded`, either because they should remain at their init value,
      or because they are missing on either side.
    match_dtype: returned pytree as leaves converted to dtype from `inited`.

  Returns:
    If successful, a new pytree which matches the structure of `init`
    but contains values from `loaded`, except for `dont_load`.

    If structures don't match and mismatches are not covered by regexes in
    `dont_load` argument, then raises an exception with more information.
  """
  if inited is None:  # A useful shortcut for example for colabs.
    return loaded

  dont_load = u.check_and_compile_patterns(dont_load)

  def should_merge(name):
    return not any(pattern.fullmatch(name) for pattern in dont_load)

  loaded_flat, _ = u.tree_flatten_with_names(loaded)
  inited_flat, _ = u.tree_flatten_with_names(inited)
  loaded_flat = {k: v for k, v in loaded_flat}
  inited_flat = {k: v for k, v in inited_flat}

  # Let's first build the pytree from all common keys.
  merged = {}
  for name, init_val in inited_flat.items():
    # param is present in both. Load or ignore it!
    if name in loaded_flat and should_merge(name):
      merged[name] = loaded_flat[name]
      if match_dtype:
        merged[name] = loaded_flat[name].astype(init_val.dtype)
    else:
      logging.info("Ignoring checkpoint and using init value for %s", name)
      merged[name] = init_val

  def pp(title, names, indent="  "):  # Just pretty-printing
    if names:
      return f"{title}:\n" + "\n".join(f"{indent}{k}" for k in sorted(names))
    else:
      return ""

  # Now, if there are keys that only exist in inited or loaded, be helpful:
  not_in_loaded = inited_flat.keys() - loaded_flat.keys()
  not_in_inited = loaded_flat.keys() - inited_flat.keys()
  logging.info(pp("Parameters in model but not in checkpoint", not_in_loaded))
  logging.info(pp("Parameters in checkpoint but not in model", not_in_inited))

  # And now see if any of them are not explicitly ignored => an error
  not_in_loaded = {k for k in not_in_loaded if should_merge(k)}
  not_in_inited = {k for k in not_in_inited if should_merge(k)}

  if not_in_loaded or not_in_inited:
    raise ValueError(
        pp("Params in checkpoint", loaded_flat.keys()) + "\n" +
        pp("Params in model (code)", inited_flat.keys()) + "\n" +
        pp("Params in model (code) but not in checkpoint and not `dont_load`ed",
           not_in_loaded, indent=" - ") + "\n" +  # Special indent for tests.
        pp("Params in checkpoint but not in model (code) and not `dont_load`ed",
           not_in_inited, indent=" + "))  # Special indent for tests.

  return u.recover_tree(merged.keys(), merged.values())


class AddPositionEmbs(nn.Module):
  """Adds positional embeddings to the inputs, supports caching for decode.

  Attributes:
    decode: whether to run in single-position autoregressive mode.
  """
  decode: bool = False

  @nn.compact
  def __call__(self, inputs, posemb):
    """Applies AddPositionEmbs module.

    Adds posemb to the inputs, supports single-position autoregressive mode.

    Args:
      inputs: input data [batch_size, seq_len, emb_dim].
      posemb: positional embeddings.

    Returns:
      output: inputs modulated by pos-embeddings [batch_size, seq_len, emb_dim].
    """
    assert inputs.ndim == 3, f"Unexpected inputs shape: {inputs.shape}"
    _, seq_len, emb_dim = inputs.shape
    pe = posemb[:, :seq_len, :]

    if self.decode:
      is_initialized = self.has_variable("cache", "cache_index")
      # We use a cache position index for tracking decoding position.
      cache_index = self.variable("cache", "cache_index",
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        # Returns posemb[0, i, :], the positional embedding for the
        # current decoding position.
        pe = jax.lax.dynamic_slice(posemb,
                                   start_indices=jnp.array((0, i, 0)),
                                   slice_sizes=(1, 1, emb_dim))
    return inputs + pe
