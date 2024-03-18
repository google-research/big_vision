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

"""Evaluator for the classfication task."""
from functools import partial, lru_cache

from big_vision import input_pipeline
import big_vision.datasets.core as ds_core
import big_vision.pp.builder as pp_builder
import big_vision.utils as u

import einops
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


def dist(student, teacher, kind, feat_axis=-1,
         epsilon=1e-12, t=1, ls=0.0, k=1):
  """Distance function used for distillation."""
  diff = student - teacher
  if kind == 'euclidean':
    return jnp.sqrt(jnp.sum(diff * diff, axis=feat_axis) + epsilon)
  elif kind == 'l2':
    return jnp.sum(diff * diff, axis=feat_axis)
  elif kind == 'hard':
    pseudolabels = jnp.argmax(teacher, feat_axis)
    pl = u.onehot(pseudolabels, teacher.shape[feat_axis])
    if ls:
      pl = (1.0 - ls) * pl + (ls / (pl.shape[-1] - 1)) * (1.0 - pl)
    return u.softmax_xent(logits=student, labels=pl,
                          reduction=False, kl=True, axis=feat_axis)
  elif kind == 'kl':
    return t**2 * u.softmax_xent(
        logits=student / t,
        labels=jax.nn.softmax(teacher / t),
        reduction=False, kl=True, axis=feat_axis)
  elif kind == 'logsoftmax_euclidean':
    logsoftmax_diff = (
        jax.nn.log_softmax(student, axis=feat_axis) -
        jax.nn.log_softmax(teacher, axis=feat_axis))
    return jnp.sqrt(
        jnp.sum(logsoftmax_diff * logsoftmax_diff, axis=feat_axis) + epsilon)
  elif kind == 'agree':
    def get_top_k(arr, k, ax):
      return jax.lax.top_k(arr.swapaxes(ax, -1), k)[1].swapaxes(ax, -1)
    return (get_top_k(student, k, feat_axis) ==
            get_top_k(teacher, 1, feat_axis)).sum(feat_axis)
  else:
    assert False, f'Unknown kind of distance {kind}.'


@lru_cache(None)
def get_dist_fn(**kw):
  return partial(dist, **kw)


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@lru_cache(None)
def get_eval_fn(student_teacher_fwd, what, mesh, distances):
  """Produces eval function, also applies pmap."""
  @partial(jax.jit, out_shardings=NamedSharding(mesh, P()))
  def _eval_fn(train_state, batch, mask):
    (_, out_s), (_, out_t) = student_teacher_fwd(train_state, batch)
    repr_s = u.tree_get(out_s, what[0])
    repr_t = u.tree_get(out_t, what[1])

    # Let's flatten any non-vectors (eg feature-maps).
    repr_s = einops.rearrange(repr_s, 'b ... -> b (...)')
    repr_t = einops.rearrange(repr_t, 'b ... -> b (...)')

    all_ds = []
    # NOTE: we're gathering and returning all ; if this becomes too slow, we
    #       can change to compute and return summary stats later on.
    for dist_fn in distances:
      ds = dist_fn(repr_s, repr_t)
      all_ds.append(ds)
    all_masks = mask
    return all_ds, all_masks

  return _eval_fn


class Evaluator:
  """Distillation distance evaluator."""

  def __init__(
      self,
      student_teacher_fwd,
      data,
      pp_fn,
      distances,
      what=('logits', 'logits'),
      *,
      devices,
      **data_kw,
  ):
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    prefetch = data_kw.pop('prefetch', 1)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True),
        pp_fn,
        num_ex_per_process=data.num_examples_per_process(),
        **data_kw,
    )
    self.data_iter = input_pipeline.start_global(self.ds, devices, prefetch)
    dist_fns = tuple(get_dist_fn(**dist) for dist in distances)
    self.dist_names = [
        '_'.join(f'{k}={v}' for k, v in dist.items()) for dist in distances
    ]
    mesh = jax.sharding.Mesh(devices, ('data',))
    self.eval_fn = get_eval_fn(student_teacher_fwd, what, mesh, dist_fns)

  def run(self, train_state):
    """Computes all metrics."""
    all_ds = [[] for _ in self.dist_names]
    for _, batch in zip(range(self.steps), self.data_iter):
      mask = batch.pop('_mask')
      batch_ds, batch_ms = self.eval_fn(train_state, batch, mask)
      # All results are a replicated array shaped as follows:
      # (local_devices, per_device_batch_size, elem_shape...)
      # with each local device's entry being identical.
      # So let's just take the first one to the host as numpy.
      batch_ms = np.array(batch_ms)
      for i, val in enumerate(batch_ds):
        all_ds[i].append(np.array(val)[batch_ms == 1])
    for name, ds in zip(self.dist_names, all_ds):
      ds = np.concatenate(ds)
      yield f'{name}/all', ds
      yield f'{name}/avg', np.mean(ds)
      yield f'{name}/min', np.min(ds)
      yield f'{name}/max', np.max(ds)
