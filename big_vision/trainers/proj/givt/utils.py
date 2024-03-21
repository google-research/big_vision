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

"""Utils for GIVT stage I and II trainers."""

from typing import Any

import jax
import jax.numpy as jnp


def unbin_depth(
    depth: jax.Array,
    *,
    min_depth: float,
    max_depth: float,
    num_bins: int,
) -> jax.Array:
  """Transform a depth map with binned values into a float-valued depth map.

  Args:
    depth: Depth map whose binned values are encoded in one-hot fashion along
      the last dimension.
    min_depth: Minimum binned depth value.
    max_depth: Maximum value of binned depth.
    num_bins: Number of depth bins.

  Returns:
    Float-valued depth map.
  """
  depth = jnp.argmax(depth, axis=-1)
  depth = depth.astype(jnp.float32) + 0.5  # Undoes floor in expectation.
  depth /= num_bins
  return depth * (max_depth - min_depth) + min_depth


def get_local_rng(
    seed: int | jax.Array,
    batch: Any,
) -> jax.Array:
  """Generate a per-image seed based on the image id or the image values.
  
  Args:
    seed: Random seed from which per-image seeds should be derived.
    batch: Pytree containing a batch of images (key "image") and optionally
      image ids (key "image/id").

  Returns:
    Array containing per-image ids.
  """
  fake_id = None
  if "image" in batch:
    fake_id = (10**6 * jax.vmap(jnp.mean)(batch["image"])).astype(jnp.int32)
  return jax.lax.scan(
      lambda k, x: (jax.random.fold_in(k, x), None),
      jax.random.PRNGKey(seed),
      batch.get("image/id", fake_id),
  )[0]

