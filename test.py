import jax
import os
import sys
import jax.numpy as jnp

import os

import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
import big_vision.utils as u

def initialize_jax():
    try:
        coordinator_address = sys.argv[2]
        port = sys.argv[4]
        num_processes = int(os.environ['SLURM_NNODES'])
        process_id = int(os.environ['SLURM_NODEID'])
        ca = f"{coordinator_address}:{port}"

         # TODO not sure how to get number of GPUs per task.
        local_device_ids = list(range(num_processes * 4))
        jax.distributed.initialize(
                # coordinator_address=ca, 
                # num_processes=int(num_processes),
                # initialization_timeout=60,
                # process_id=process_id,
                local_device_ids=local_device_ids, # This needs to be set as default is broken for slurm
                )
    except Exception as ex:
        print("Exception: ")
        print(ex)

initialize_jax()

print("SLURM_NTASKS_PER_GPU", os.environ.get("SLURM_NTASKS_PER_GPU", "not-set"))
print("SLURM_GPUS", os.environ.get("SLURM_GPUS", "not-set"))

# This should subset of the jax.devices()
print(jax.local_devices())
print(jax.devices())


n = jax.device_count()
mesh_shape = (n,)
device_mesh = np.array(jax.devices()).reshape(mesh_shape)
mesh = Mesh(device_mesh, ("mdl",))


global_devices = np.asarray(jax.devices()).reshape(-1)

def _shard(x):
    mesh = jax.sharding.Mesh(global_devices, ("devices",))
    sharding = jax.sharding.NamedSharding(mesh, P("devices"))
    local_ds = mesh.local_devices

    print(f"Sharding array {x} - {local_ds=} and {global_devices=}")

    x = np.asarray(memoryview(x))  # No-copy: http://(internal link)
    xs = jax.device_put(np.split(x, len(local_ds), axis=0), local_ds)

    global_shape = (x.shape[0] * jax.process_count(), *x.shape[1:])
    return jax.make_array_from_single_device_arrays(global_shape, sharding, xs)


def z(x):
    return jnp.sum(x)

def g(x):
    return x + 1, jnp.sum(x)

process_id = int(os.environ['SLURM_NODEID'])
x = np.arange(process_id*8, (process_id+1)*8)
print("Input: ", x)
x = _shard(x)

# print("Sharding: ", x.sharding) #, x.addressable_shards)
# print([(x.data, x.device) for x in x.addressable_shards])


y, s = jax.jit(g)(x)
print("sum: ", s, s.shape)

num_processes = int(os.environ['SLURM_NNODES'])
assert jax.device_get(s) == sum(range(8 * num_processes)), s

y = jax.experimental.multihost_utils.process_allgather(y)
print("Y:", y, y.shape)

assert y.tolist() == list(range(1, 8 * num_processes + 1)), y

print("Done!")