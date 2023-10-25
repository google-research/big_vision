# Copyright 2023 Big Vision Authors.
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

import jax
import numpy as np


NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def _replicated(mesh):
  return NamedSharding(mesh, P())


def _shard_along_axis(mesh, i, axis_name):
  return NamedSharding(mesh, P(*((None,) * i + (axis_name,))))


def infer_sharding(params, mesh, axis_name, strategy, extra_strategy_args):
  """Infers `params` sharding based on strategy.

  Args:
    params: a pytree of arrays.
    mesh: jax device mesh.
    axis_name: a device axis name (from mesh) to use for sharding.
    strategy: sharding strategy.
    extra_strategy_args: additional args to be passed to the strategy.

  Returns:
    A pytree with shardings, that has the same shape as the `tree` argument.
  """
  sharding_fn = {
      "replicated": replicated,
      "fully_sharded": fully_sharded,
  }[strategy]
  return sharding_fn(params, mesh, axis_name, **extra_strategy_args)


def replicated(params, mesh, axis_name):
  del axis_name
  return jax.tree_map(lambda _: NamedSharding(mesh, P()), params)


def fully_sharded(params, mesh, axis_name, too_small_to_shard_thr=2 ** 18):
  """Shards all large tensors along the largest dim, otherwise replicates."""
  idx = mesh.axis_names.index(axis_name)
  axis_size = np.shape(mesh.devices)[idx]
  def _get_spec(x):
    shape = x.shape

    # Do not bother with partioning <=1 MB arrays.
    if np.prod(shape) <= too_small_to_shard_thr:
      return _replicated(mesh)

    # Otherwise parition along largest axis.
    idx = np.argsort(shape)[::-1]
    for i in idx:
      if shape[i] % axis_size == 0:
        return _shard_along_axis(mesh, i, axis_name)

    # Array is not evenly shardable, just replicate it then.
    return _replicated(mesh)

  return jax.tree_map(_get_spec, params)

