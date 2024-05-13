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

"""Gradient transformations and other optax utilities."""

import operator
import big_vision.utils as u
import jax
import jax.numpy as jnp
import optax


def find_states(opt_state, cls):
  leaves = jax.tree.leaves(
      opt_state, is_leaf=lambda node: isinstance(node, cls))
  return [leaf for leaf in leaves if isinstance(leaf, cls)]


def get_count(opt_state, jittable=False):
  """Returns `ScaleByScheduleState.count` from `opt_state` as an integer."""
  counts = [
      state.count
      for state in find_states(opt_state, optax.ScaleByScheduleState)
  ]
  if jittable:
    return counts[0]
  else:
    counts = {int(c) for c in counts}
    assert len(counts) == 1, f"Expected exactly 1 ScaleByScheduleState:{counts}"
    return next(iter(counts))


def replace_frozen(schedule, pytree, replacement, log=None):
  """Replaces values matching frozen params in `pytree` with `replacement`."""
  if not isinstance(schedule, (list, tuple)):
    return pytree
  masks, scheds = _make_mask_trees(pytree, schedule, log=log)
  frozen_mask, _, _ = _split_frozen(masks, scheds)
  return jax.tree.map(
      lambda v, f: replacement if f else v, pytree, frozen_mask)


def make(config, params, *, sched_kw):
  """Returns gradient transform and learning rate functions."""

  # Global schedule. No schedule means frozen.
  schedule = config.get("schedule", {})
  if not isinstance(schedule, (tuple, list)):
    schedule = [(".*", schedule)]
  masks, scheds = _make_mask_trees(params, schedule, "config.schedule")
  frozen_mask, masks, scheds = _split_frozen(masks, scheds)
  not_frozen_mask = jax.tree.map(operator.not_, frozen_mask)
  def create_schedule(mult=1.0, **kw):
    assert "base" not in kw, kw
    return u.create_learning_rate_schedule(base=mult, **kw)
  schedule_fns = [create_schedule(**sched_kw, **sched) for sched in scheds]
  schedule_txs = [
      optax.masked(optax.scale_by_schedule(schedule_fn), mask)
      for schedule_fn, mask in zip(schedule_fns, masks)
  ] + [
      # Removes weight decay updates. Note that weight decay already has an
      # independent mask (which cannot be combined easily with a second mask),
      # so instead we multiply updates for frozen params with zero.
      optax.masked(optax.set_to_zero(), frozen_mask)
  ]

  # Gradient clipping.
  grad_clip_norm_tx = (
      optax.masked(optax.clip_by_global_norm(config.grad_clip_norm),
                   not_frozen_mask)
      if config.get("grad_clip_norm") else optax.identity())

  # Optimizer updates.
  tx_func = operator.attrgetter(config.optax_name)(optax)
  opt_txs = [optax.masked(tx_func(**config.get("optax", {})), not_frozen_mask)]
  assert "optim" not in config, "Deprecated option, use config.optax."

  # Learning rate multipliers. Defaults to 1.0.
  lr_mult_txs = [optax.scale(config.lr)]
  if config.get("lr_mults"):
    masks, mults = _make_mask_trees(params, config.lr_mults, "config.lr_mults")
    assert all(mult > 0 for mult in mults), (
        f"Use schedule=None for parameter freezing instead of lr_mults={mults}")
    lr_mult_txs += [
        optax.masked(optax.scale(mult), mask)
        for mult, mask in zip(mults, masks)
    ]

  # Weight decay. Defaults to 0.0.
  # Weight decay is not gradient-based but instead uses "params side-input".
  # Hence, weight decay is additive and independent of previous gradient-based
  # updates.
  assert "weight_decay" not in config, "Deprecated option. Use wd and schedule."
  assert config.get("weight_decay_decouple", True), (
      "Coupled weight decay not supported anymore.")
  if config.get("wd"):
    wd_mults = config.get("wd_mults", [(".*/kernel$", 1.0)])
    masks, mults = _make_mask_trees(params, wd_mults, "config.wd_mults")
    weight_decay_txs = [
        optax.add_decayed_weights(config.wd * mult, mask)
        for mult, mask in zip(mults, masks)
    ]
  else:
    weight_decay_txs = []

  # Combine gradient updates and learning rate schedules.
  return optax.chain(
      grad_clip_norm_tx,
      *opt_txs,
      *lr_mult_txs,
      *weight_decay_txs,
      *schedule_txs,
      optax.scale(-1.0)), schedule_fns


def _make_mask_trees(params, patterns_values, log):
  patterns, values = zip(*patterns_values)
  masks = u.make_mask_trees(params, patterns, log=log)
  return masks, values


def _split_frozen(masks, scheds):
  """Computes `frozen_mask` and updates `masks` and `scheds`."""
  # Specifying `None` as a scheduler freezes params.
  all_false = jax.tree.map(lambda *bools: not any(bools), *masks)
  not_covered = [k for k, v in u.tree_flatten_with_names(all_false)[0] if v]
  assert not not_covered, (
      f"All params must be covered (use `None` for freezing): {not_covered}")
  frozen_masks = [
      mask for mask, sched in zip(masks, scheds) if sched is None]
  frozen_mask = jax.tree.map(
      lambda *bools: any(bools), *frozen_masks,
      all_false)  # `all_false` is required when `frozen_masks==[]`.
  masks, scheds = zip(*(
      (mask, sched) for mask, sched in zip(masks, scheds) if sched is not None))
  return frozen_mask, masks, scheds


############ Custom BigVision optimizers #######################################
# Currently there's only one custom optimizer and we don't foresee new ones in
# the near future, we opt not to create a new optimizer folder/module for just
# one isolated case. If there will be more optimizers, we can consider moving
# them into individual files in a subfolder.


# A dummy object to allow for foo.bar access syntax, see
# https://stackoverflow.com/a/19476841/2366315
optax.big_vision = type("", (), {})()


def scale_by_adafactor(min_dim_size_to_factor=32,
                       decay_rate=0.8, decay_offset=0,
                       beta2_cap=0.999,
                       clipping_threshold=None,
                       momentum=0.9, dtype_momentum=jnp.bfloat16,
                       eps=1e-30):
  """The BigVision variant of Adafactor optimizer."""

  def _decay_rate_pow(i, exponent):
    """Second-order moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return jnp.minimum(beta2_cap, 1.0 - t**(-exponent))

  scale_by_rms = optax.scale_by_factored_rms(
      factored=True,
      decay_rate=decay_rate,
      step_offset=decay_offset,
      min_dim_size_to_factor=min_dim_size_to_factor,
      epsilon=eps,
      decay_rate_fn=_decay_rate_pow)

  clip = (optax.clip_by_block_rms(clipping_threshold) if clipping_threshold
          else optax.identity())

  mom = (optax.ema(momentum, debias=False, accumulator_dtype=dtype_momentum)
         if momentum else optax.identity())

  return optax.chain(scale_by_rms, clip, mom)

optax.big_vision.scale_by_adafactor = scale_by_adafactor  # pytype: disable=module-attr


# A few more aliases we use frequently:
def momentum_hp(momentum=0.9, dtype=jnp.bfloat16, nesterov=False):
  """SGD-Momentum with half-precision accumulator."""
  return optax.trace(decay=momentum, accumulator_dtype=dtype, nesterov=nesterov)

optax.big_vision.momentum_hp = momentum_hp  # pytype: disable=module-attr
optax.big_vision.sgd = optax.identity  # pytype: disable=module-attr
