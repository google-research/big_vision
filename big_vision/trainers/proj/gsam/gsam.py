# Copyright 2022 Big Vision Authors.
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
'''This file provides jax implementation of GSAM.'''

import jax
import jax.numpy as jnp

def dual_vector(y):
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient, gradient_norm

def gsam_gradient(loss_fn, params, inputs, targets,
                  rho_max, rho_min, alpha, lr, lr_max, lr_min, eps=1e-12,
                  adaptive_perturbation=False, minimize_fp=True):
  """
  Get the GSAM gradient (https://openreview.net/pdf?id=edONMAnhLu-).
  Args:
    loss_fn: the loss function.
    params: the model weights.
    inputs: the inputs to the loss function.
    targets: the targets to the loss function.
    rho_max: the maximum rho value for perturbation of weights.
    rho_min: the minimum rho value for perturbation of weights.
    alpha: the alpha value for the rho schedule, see Algorithm 1 in the paper.
    lr: current learning rate.
    lr_max: the maximum learning rate.
    lr_min: the minimum learning rate.
    eps: the epsilon value for numerical stability.
    adaptive_perturbation: if False, same perturbation as SAM,
        treat all parameters as a single vector,
        perturbation norm is calculated as the norm of the whole vector;
        If True, perturbation norm is proportional to parameter norm,
        this stabilizes training when different layers have weights
        of different scales.
        Emprically, setting it to True can handle 10x larger rho than
        setting it to False.
    minimize_fp: if True, min(f_p, h), original GSAM;
        if False, min(f, h), where f is the clean loss.
        f_p is the perturbed loss, h is the surrogate gap.
        If True, training dynamics is closer to SAM than conventional training,
        you might observe several loss spikes during training.
        If False, the training dynamics is closer to conventional training,
        and is often more stable (fewer loss spikes) during training.
  Returns:
    l_clean: the loss function value.
    g_gsam: the GSAM gradient. g_gsam is not averaged across workers,
        need to call "jax.lax.pmean" to average.

  Note:
    Setting `rho_max=rho_min` and `alpha=0` reduces GSAM to SAM.
  """
  l_clean, g_clean = jax.value_and_grad(loss_fn)(params, inputs, targets)
  g_clean_normalized, g_clean_length = dual_vector(g_clean)

  if lr_max == lr_min:
    sam_rho = rho_max
  else:
    sam_rho = rho_min + (rho_max - rho_min) * (lr - lr_min) / (lr_max - lr_min)

  # Per-worker perturbation.
  if adaptive_perturbation:
    param_sam = jax.tree_map(lambda a, b: a + \
        jnp.abs(a) * sam_rho * b / (g_clean_length + eps), params, g_clean)
  else:
    param_sam = jax.tree_map(lambda a, b: a + \
        sam_rho * b / (g_clean_length + eps), params, g_clean)

  # Get gradients at perturbed weights.
  _, g_robust = jax.value_and_grad(loss_fn)(param_sam, inputs, targets)

  # Decompose gradients.
  g_clean_flatten, _ = jax.tree_util.tree_flatten(g_clean)
  g_robust_flatten, _ = jax.tree_util.tree_flatten(g_robust)

  if minimize_fp:
    # Decompose g_clean onto parallel and vertical to g_robust.
    g_robust_normalized, _ = dual_vector(g_robust)
    g_robust_normalized_flatten, _ = jax.tree_util.tree_flatten(
        g_robust_normalized)

    g_clean_projection_norm = sum(jnp.vdot(p, q) for (p,q) in
        zip(g_robust_normalized_flatten, g_clean_flatten))
    g_clean_residual = jax.tree_map(lambda a, b:
        a - g_clean_projection_norm * b, g_clean, g_robust_normalized)

    # Get GSAM gradient.
    g_gsam = jax.tree_map(lambda a, b: a - b * alpha,
        g_robust, g_clean_residual)
  else:
    # Decompose g_robust onto parallel and vertical to g_clean.
    g_clean_normalized, g_clean_length = dual_vector(g_clean)
    g_clean_normalized_flatten, _ = jax.tree_util.tree_flatten(
        g_clean_normalized)

    g_robust_projection_norm = sum(jnp.vdot(p, q) for (p,q) in
        zip(g_clean_normalized_flatten, g_robust_flatten))
    g_robust_residual = jax.tree_map(lambda a, b:
        a - g_robust_projection_norm * b, g_robust, g_clean_normalized)

    # Get GSAM gradient.
    g_gsam = jax.tree_map(lambda a, b: a + b * alpha,
        g_clean, g_robust_residual)

  # Always return the clean loss (rather than the perturbed loss).
  return l_clean, g_gsam
