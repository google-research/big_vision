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

"""Utils for few-shot evaluation."""
# pylint: disable=consider-using-from-import

import functools

import big_vision.datasets.core as ds_core
import big_vision.input_pipeline as input_pipeline
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding as Sharding
from jax.sharding import PartitionSpec as P
import numpy as np

BIAS_CONSTANT = 100.0

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


# Setup function for few-shot regression on CPU to avoid "polluting" the TPU.
@u.jit_cpu(static_argnums=(2,))
def _precompute_cache(x, y, num_classes):
  """Cache quantities to speed-up the computation of L2-regularized least-sq."""
  # Whiten
  mean = jnp.mean(x, axis=0, keepdims=True)
  std = jnp.std(x, axis=0, keepdims=True) + 1e-5
  x = (x - mean) / std

  # Add a constant feature for the bias, large so it's almost unregularized:
  x = jnp.pad(x, ((0, 0), (0, 1)), constant_values=BIAS_CONSTANT)

  # To one-hot representation rescaled into {-1, 1}
  y = 2.0 * jax.nn.one_hot(y, num_classes) - 1.0

  num_points, dim = x.shape
  # Let N be the number of points, D the dimension and C the number of classes.
  # We have x of shape (N, D) and y of shape (N, C).
  # For least-squares, we can compute
  #
  #   (A) when N >= D, (x^T x + l2 Id)^{-1} x^T y
  #   (B) when D > N, x^T  (x x^T + l2 Id)^{-1} y
  #
  # We pre-compute the eigen-decomposition of either x^T x or x x^T which
  # becomes q diag(eigs) q^T with q unitary matrix either (D, D) or (N, N)
  # and eigs a vector (D,) or (N,).
  #
  # For any l2 > 0, we can compute (x^T x + l2 Id)^{-1} or (x x^T + l2 Id)^{-1}
  # by simply computing q (diag(eigs) + l2 Id)^{-1} q^T.
  # (SVD would be more natural here, but it proved slower, so we use eigh)
  #
  # Both cases (A) and (B) can be viewed as lhs (diag(eigs) + l2 Id)^{-1} rhs,
  # where lhs/rhs are pre-computed left/right-hand sides to specify.
  #
  # Detailed evaluation in terms of time and fewshot metrics can be found in
  # (internal link)
  #
  # Implemented by Rodolphe Jenatton.
  if num_points >= dim:
    eigs, q = jnp.linalg.eigh(x.T @ x)
    rhs = q.T @ (x.T @ y)
    lhs = q
  else:
    eigs, q = jnp.linalg.eigh(x @ x.T)
    rhs = q.T @ y
    lhs = x.T @ q

  cache = {
      "eigs": eigs,
      "rhs": rhs,
      "lhs": lhs,
      "mean": mean,
      "std": std
  }
  return cache


@u.jit_cpu()
def _eig_fewshot_acc_fn(cache, x_test, y_test, l2_reg):
  """Computes (x,y) linear regression accuracy on (x_test, y_test)."""

  x_test = (x_test - cache["mean"]) / cache["std"]
  x_test = jnp.pad(x_test, ((0, 0), (0, 1)), constant_values=BIAS_CONSTANT)

  rhs = cache["rhs"]
  lhs = cache["lhs"]
  eigs = cache["eigs"]

  # See comments in _precompute_cache for context about the formula.
  scaling = 1.0 / (eigs + l2_reg * jnp.ones_like(eigs))
  scaling = scaling.reshape((1, -1))
  w = (lhs * scaling) @ rhs
  # Predict test-set values and measure their accuracy
  preds = jnp.argmax(x_test @ w, axis=1)
  return jnp.mean(preds == y_test)


class Evaluator:
  """Class for few-shot evaluation."""

  def __init__(self, predict_fn, batch_size,
               datasets, shots, l2_reg,
               pp_train, pp_eval, display_first,
               representation_layer=None, num_seeds=3,
               label_key="label", mask_key="_mask", *,
               devices):
    self.datasets = datasets
    self.shots = shots
    self.l2_reg = l2_reg
    self.batch_size = batch_size
    self.pp_tr = pp_train
    self.pp_te = pp_eval
    self.display_first = display_first
    self._datasets = {}  # Cache for tfds data. Persists while object is alive.
    self._repr = {}  # Cache for precomputed repr. Persists within the run call.
    self.num_seeds = num_seeds
    self.label_key = label_key
    self.mask_key = mask_key

    self.devices = devices
    self.mesh = jax.sharding.Mesh(devices, ("devices",))
    self.repr_fn = self.get_representation_fn(
        predict_fn, representation_layer)

  def get_representation_fn(self, predict_fn, representation_layer):
    # `out_shardings=Sharding(self.mesh, P())` will "all_gather" the outputs.
    @functools.partial(jax.jit, out_shardings=Sharding(self.mesh, P()))
    def _repr_fn(train_state, batch, labels, mask):
      zimg, *_, out = predict_fn(train_state, batch)
      if representation_layer is not None:
        rep = u.tree_get(out, representation_layer)
      else:
        rep = zimg
      return rep, labels, mask
    return _repr_fn

  # Setup input pipeline.
  def _get_dataset(self, dataset, train_split, test_split):
    """Lazy-loads given dataset."""
    key = (dataset, train_split, test_split)
    try:
      return self._datasets[key]
    except KeyError:
      # NOTE: only supporting TFDS data for now for bwd compat/lazyness.
      train_data = ds_core.get(name=dataset, split=train_split)
      train_ds, batches_tr = input_pipeline.make_for_inference(
          train_data.get_tfdata(ordered=True),
          num_ex_per_process=train_data.num_examples_per_process(),
          batch_size=self.batch_size,
          preprocess_fn=pp_builder.get_preprocess_fn(self.pp_tr))
      test_data = ds_core.get(name=dataset, split=test_split)
      test_ds, batches_te = input_pipeline.make_for_inference(
          test_data.get_tfdata(ordered=True),
          num_ex_per_process=test_data.num_examples_per_process(),
          batch_size=self.batch_size,
          preprocess_fn=pp_builder.get_preprocess_fn(self.pp_te))
      num_classes = train_data.builder.info.features["label"].num_classes
      return self._datasets.setdefault(
          key, (train_ds, batches_tr, test_ds, batches_te, num_classes))

  def _get_repr(self, params, data, steps):
    """Compute representation for the whole dataset."""
    pre_logits_list = []
    labels_list = []
    for batch, _ in zip(
        input_pipeline.start_global(data, self.devices, 0), range(steps)):
      labels, mask = batch.pop(self.label_key), batch.pop(self.mask_key)
      pre_logits, labels, mask = jax.device_get(self.repr_fn(
          params, batch, labels, mask))
      mask = mask.astype(bool)
      pre_logits_list.append(pre_logits[mask])
      labels_list.append(labels[mask])
    pre_logits = np.concatenate(pre_logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return pre_logits, labels

  def compute_fewshot_metrics(self, train_state, seed,
                              dataset, train_split, test_split):
    """Compute few-shot metrics on one dataset."""
    if dataset in self._repr:
      repr_train, labels_train, repr_test, labels_test, num_classes = (
          self._repr[dataset])
    else:
      train_ds, steps_tr, test_ds, steps_te, num_classes = self._get_dataset(
          dataset, train_split, test_split)
      repr_train, labels_train = self._get_repr(train_state, train_ds, steps_tr)
      repr_test, labels_test = self._get_repr(train_state, test_ds, steps_te)
      self._repr[dataset] = (repr_train, labels_train,
                             repr_test, labels_test,
                             num_classes)

    # Collect where we have samples of which classes.
    rng = np.random.default_rng(seed)
    class_indices = [rng.permutation(np.where(labels_train == cls_i)[0])
                     for cls_i in range(num_classes)]

    results = {}
    for shots in self.shots:
      all_idx = [indices[:shots] for indices in class_indices]
      all_idx = np.concatenate(all_idx, axis=0)
      x = u.put_cpu(repr_train[all_idx])
      y = u.put_cpu(labels_train[all_idx])
      repr_test, labels_test = u.put_cpu((repr_test, labels_test))

      # Note the code is optimized to solve multiple LSR tasks for changing l2
      # strength, even though we currently used the fixed l2_reg constant.
      cache = _precompute_cache(x, y, num_classes)
      acc = _eig_fewshot_acc_fn(
          cache, repr_test, labels_test, u.put_cpu(self.l2_reg))
      results[shots] = jax.device_get(acc)

    return results

  def run(self, train_state):
    """New API executed in terms of old API."""
    self._repr = {}
    for seed in range(self.num_seeds):
      for name, dataset_args in self.datasets.items():
        result = self.compute_fewshot_metrics(train_state, seed, *dataset_args)
        for shots, v in result.items():
          prefix = "a/" if (name, shots) in self.display_first else "z/"
          suffix = f"-seed-{seed}"
          yield f"{prefix}{name}_{shots}shot{suffix}", v
