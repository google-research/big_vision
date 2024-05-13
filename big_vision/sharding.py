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

"""Big vision sharding utilities."""

from absl import logging

from big_vision.pp.registry import Registry
import big_vision.utils as u
import flax.linen as nn
import jax
import numpy as np


NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def _replicated(mesh):
  return NamedSharding(mesh, P())


def _shard_along_axis(mesh, i, axis_name):
  return NamedSharding(mesh, P(*((None,) * i + (axis_name,))))


def infer_sharding(params, strategy, mesh):
  """Infers `params` sharding based on strategy.

  Args:
    params: a pytree of arrays.
    strategy: sharding strategy.
    mesh: jax device mesh.

  Returns:
    A pytree with shardings, that has the same shape as the `tree` argument.
  """
  patterns, tactics = zip(*strategy)

  x_with_names, tree_def = u.tree_flatten_with_names(params)
  names = tree_def.unflatten(list(zip(*x_with_names))[0])

  # Follows big_vision conventions: each variable is matched at most once,
  # early patterns get matching priority.
  mask_trees = u.make_mask_trees(params, patterns)

  specs = jax.tree.map(lambda x: (None,) * x.ndim, params)

  for mask_tree, tactic in zip(mask_trees, tactics):
    for op_str in tactic.split("|"):
      op = Registry.lookup(f"shardings.{op_str}")()
      specs = jax.tree.map(
          lambda x, n, match, spec, op=op: op(spec, mesh, n, x)
          if match else spec,
          params, names, mask_tree, specs,
          is_leaf=lambda v: isinstance(v, nn.Partitioned))

  # Two-level tree_map to prevent it from doing traversal inside the spec.
  specs = jax.tree.map(lambda _, spec: P(*spec), nn.unbox(params), specs)
  return jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs)


# Sharding rules
#
# Each rule needs to be added to the registry, can accept custom args, and
# returns a function that updates the current spec. The arguments are:
# 1. Variable name
# 2. Variable itself (or placeholder with .shape and .dtype properties)
# 3. The current sharing spec.


@Registry.register("shardings.replicate")
def replicate():
  """Full replication sharding rule.

  Note full replication is deafult, so this can be skipped and useful to
  explicitly state in the config that certrain parameters are replicated.
  TODO: can be generalized to support replication over a sub-mesh.

  Returns:
    A function that updates the sharding spec.
  """
  def _update_spec(cur_spec, mesh, name, x):
    del x, mesh
    if not all(axis is None for axis in cur_spec):
      raise ValueError(f"Inconsistent sharding instructions: "
                       f"parameter {name} has spec {cur_spec}, "
                       f"so it can't be fully replicated.")
    return cur_spec
  return _update_spec


@Registry.register("shardings.fsdp")
def fsdp(axis, min_size_to_shard_mb=4):
  """FSDP sharding rule.

  Shards the largest dimension that is not sharded already and is divisible
  by the total device count.

  Args:
    axis: mesh axis name for FSDP.
    min_size_to_shard_mb: minimal tensor size to bother with sharding.

  Returns:
    A function that updates the sharding spec.
  """
  def _update_spec(cur_spec, mesh, name, x):
    shape = x.shape
    axis_size = mesh.shape[axis]

    if np.prod(shape) * x.dtype.itemsize <= min_size_to_shard_mb * (2 ** 20):
      return cur_spec

    # Partition along largest axis that is divisible and not taken.
    idx = np.argsort(shape)[::-1]
    for i in idx:
      if shape[i] % axis_size == 0:
        if cur_spec[i] is None:
          return cur_spec[:i] + (axis,) + cur_spec[i+1:]

    logging.info("Failed to apply `fsdp` rule to the parameter %s:%s, as all "
                 "its dimensions are not divisible by the requested axis: "
                 "%s:%i, or already occupied by other sharding rules: %s",
                 name, shape, axis, axis_size, cur_spec)
    return cur_spec
  return _update_spec


@Registry.register("shardings.logical_partitioning")
def logical_partitioning():
  """Manual sharding based on Flax's logical partitioning annotations.

  Uses logical sharding annotations added in model code with
  `nn.with_logical_partitioning`.  Respects logical to mesh name mapping rules
  (typically defined in the dynamic context using
  `with nn.logical_axis_rules(rules): ...`).

  Returns:
    A function that outputs the sharding spec of `nn.LogicallyPartitioned` boxed
    specs.
  """
  def _update_spec(cur_spec, mesh, name, x):
    del x, name, mesh
    if isinstance(cur_spec, nn.LogicallyPartitioned):
      return nn.logical_to_mesh_axes(cur_spec.names)
    return cur_spec
  return _update_spec
