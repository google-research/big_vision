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

"""Utility functions."""

import jax
from jax.experimental import shard_map
from jax.interpreters import pxla


P = jax.sharding.PartitionSpec


def batch_shmap(fn, *args, **kwargs):
  """Shard map to map along the data dimension w/o triggering communication."""

  mesh = pxla.thread_resources.env.physical_mesh
  if not mesh.empty:
    devices_flat = mesh.devices.flatten()
    mesh_flat = jax.sharding.Mesh(devices_flat, ("data",))
    fn = shard_map.shard_map(
        fn,
        mesh=mesh_flat,
        in_specs=P("data"), out_specs=P("data"), check_rep=True)
  return fn(*args, **kwargs)


def subsample_batch(x, subsample: int):
  """Shard map to subsample the data dimension w/o triggering communication."""
  fn = lambda x: jax.tree.map(lambda xx: xx[::subsample], x)
  return batch_shmap(fn, x) if subsample > 1 else x
