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

"""Tests for optax."""

from absl.testing import absltest
from absl.testing import parameterized
from big_vision import optax as bv_optax
import chex
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


class OptaxTest(parameterized.TestCase):

  def test_get_count(self):
    params = jax.tree.map(jnp.array, {"a": 1.})
    tx = optax.masked(
        optax.scale_by_schedule(lambda step: step),
        {"a": True},
    )
    opt_state = tx.init(params)
    self.assertEqual(bv_optax.get_count(opt_state), 0)
    _, opt_state = tx.update(params, opt_state)
    self.assertEqual(bv_optax.get_count(opt_state), 1)

  def test_split_frozen(self):
    params = jax.tree.map(jnp.array, {
        "Dense_0": {"kernel": 1., "bias": 2.},
    })  # pyformat: disable
    sched1 = dict(decay_type="cosine")
    sched2 = dict(decay_type="linear")
    schedule = [
        (".*/kernel", sched1),
        (".*/bias", sched2),
    ]
    masks, scheds = bv_optax._make_mask_trees(params, schedule, log="schedule")
    frozen_mask, masks, scheds = bv_optax._split_frozen(masks, scheds)
    chex.assert_trees_all_equal(
        frozen_mask,
        {"Dense_0": {"kernel": False, "bias": False}},
    )  # pyformat: disable
    chex.assert_trees_all_equal(
        masks,
        (
            {"Dense_0": {"kernel": True, "bias": False}},
            {"Dense_0": {"kernel": False, "bias": True}},
        ),
    )  # pyformat: disable
    self.assertEqual(scheds, (sched1, sched2))
    # freeze some
    schedule = [
        (".*/bias", None),
        ("Dense_0/.*", sched1),
        (".*", None),
    ]
    masks, scheds = bv_optax._make_mask_trees(params, schedule, log="schedule")
    frozen_mask, masks, scheds = bv_optax._split_frozen(masks, scheds)
    chex.assert_trees_all_equal(
        frozen_mask,
        {"Dense_0": {"kernel": False, "bias": True}},
    )  # pyformat: disable
    chex.assert_trees_all_equal(
        masks,
        ({"Dense_0": {"kernel": True, "bias": False}},),
    )  # pyformat: disable
    self.assertEqual(scheds, (sched1,))
    # does not cover all params - fails
    schedule = [
        (".*/kernel", None),
    ]
    masks, scheds = bv_optax._make_mask_trees(params, schedule, log="schedule")
    with self.assertRaisesRegex(AssertionError, "All params must be covered"):
      _ = bv_optax._split_frozen(masks, scheds)

  def test_replace_frozen(self):
    params = jax.tree.map(jnp.array, {
        "Dense_0": {"kernel": 1., "bias": 2.},
    })  # pyformat: disable
    schedule = [
        (".*/kernel", {}),
        (".*", None),
    ]
    chex.assert_trees_all_equal(
        bv_optax.replace_frozen(schedule, params, 0.),
        {"Dense_0": {"kernel": 1., "bias": 0.}},
    )  # pyformat: disable

  def test_make_simple(self):
    params = jax.tree.map(jnp.array, {
        "Dense_0": {"kernel": 1., "bias": 2.},
    })  # pyformat: disable

    config = ml_collections.ConfigDict()
    config.lr = 0.01
    config.schedule = dict(decay_type="linear")
    config.optax_name = "scale"
    config.optax = ml_collections.ConfigDict()
    g_scale = 0.5
    config.optax.step_size = g_scale

    total_steps = 10
    sched_kw = dict(global_batch_size=1, total_steps=total_steps)
    tx, (schedule_fn,) = bv_optax.make(config, params, sched_kw=sched_kw)
    opt_state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    for step in range(total_steps):
      updates, opt_state = tx.update(grads, opt_state)
      self.assertEqual(bv_optax.get_count(opt_state), step + 1)
      sched = schedule_fn(step)
      np.testing.assert_almost_equal(
          sched, 1.0 / total_steps * (total_steps - step))
      make_tx = lambda sched: lambda g: -sched * config.lr * g_scale * g
      chex.assert_trees_all_close(updates, jax.tree.map(make_tx(sched), grads))

  def test_make_wd(self):
    params = jax.tree.map(jnp.array, {
        "Dense_0": {"kernel": 1., "bias": 2., "other": 3.},
    })  # pyformat: disable
    wds = jax.tree.map(jnp.array, {
        "Dense_0": {"kernel": 2e-3, "bias": 5e-4, "other": 0.},
    })  # pyformat: disable

    config = ml_collections.ConfigDict()
    config.lr = 0.01
    config.wd = 1e-3
    config.wd_mults = [
        (".*/kernel", 2.0),
        (".*/bias", 0.5),
    ]
    config.schedule = dict(decay_type="linear")
    config.optax_name = "scale"
    config.optax = ml_collections.ConfigDict()
    g_scale = 0.5
    config.optax.step_size = g_scale

    total_steps = 10
    sched_kw = dict(global_batch_size=1, total_steps=total_steps)
    tx, (sched_fn,) = bv_optax.make(config, params, sched_kw=sched_kw)
    opt_state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    for step in range(total_steps):
      updates, opt_state = tx.update(grads, opt_state, params)
      self.assertEqual(bv_optax.get_count(opt_state), step + 1)
      sched = sched_fn(step)
      np.testing.assert_almost_equal(
          sched, 1.0 / total_steps * (total_steps - step))

      def make_tx(sched):
        def inner(p, g, wd):
          return -sched * (config.lr * g_scale * g + p * wd)
        return inner

      chex.assert_trees_all_close(
          updates, jax.tree.map(make_tx(sched), params, grads, wds))

  def test_make_clip_norm(self):
    params = jax.tree.map(jnp.array, {
        "Dense_0": {"kernel": 1., "bias": 2., "other": 3.},
    })  # pyformat: disable

    config = ml_collections.ConfigDict()
    config.lr = 0.01
    config.schedule = dict(decay_type="linear")
    config.optax_name = "scale"
    config.grad_clip_norm = 1.0
    config.optax = ml_collections.ConfigDict()
    g_scale = 0.5
    config.optax.step_size = g_scale

    total_steps = 10
    sched_kw = dict(global_batch_size=1, total_steps=total_steps)
    tx, (sched_fn,) = bv_optax.make(config, params, sched_kw=sched_kw)
    opt_state = tx.init(params)

    grads = jax.tree.map(jnp.ones_like, params)
    gflat = jax.tree.leaves(grads)
    l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in gflat]))
    grad_clip_factor = jnp.minimum(1.0, config.grad_clip_norm / l2_g)
    grads_scaled = jax.tree.map(lambda p: grad_clip_factor * p, grads)

    for step in range(total_steps):
      updates, opt_state = tx.update(grads, opt_state)
      self.assertEqual(bv_optax.get_count(opt_state), step + 1)
      sched = sched_fn(step)
      np.testing.assert_almost_equal(
          sched, 1.0 / total_steps * (total_steps - step))
      make_tx = lambda sched: lambda g: -sched * config.lr * g_scale * g
      chex.assert_trees_all_close(updates,
                                  jax.tree.map(make_tx(sched), grads_scaled))

  def test_make_multi(self):
    params = jax.tree.map(
        jnp.array, {
            "Dense_0": {"kernel": 1.0, "bias": 2.0, "other": 3.0},
            "Dense_1": {"kernel": 4.0, "bias": 5.0, "other": 6.0},
            "Dense_2": {"kernel": 7.0, "bias": 8.0, "other": 9.0},
            "Dense_3": {"kernel": 10., "bias": 11., "other": 12.},
        })  # pyformat: disable

    # Manually specify lr + wd for computing expected values.
    lrb = 0.01
    lr1 = 2.0
    lr2 = 0.5
    lr_mults = {
        "Dense_0": {"kernel": lr1, "bias": lr1, "other": lr1},
        "Dense_1": {"kernel": lr2, "bias": lr2, "other": lr2},
        "Dense_2": {"kernel": 1.0, "bias": 1.0, "other": 1.0},
        "Dense_3": {"kernel": 1.0, "bias": 1.0, "other": 1.0},
    }  # pyformat: disable
    wdb = 1e-3
    wd1 = 10.0
    wd2 = 0.1
    wds = jax.tree.map(
        jnp.array, {
            "Dense_0": {"kernel": wd1 * wdb, "bias": wd2 * wdb, "other": 0.},
            "Dense_1": {"kernel": wd1 * wdb, "bias": wd2 * wdb, "other": 0.},
            "Dense_2": {"kernel": wd1 * wdb, "bias": wd2 * wdb, "other": 0.},
            "Dense_3": {"kernel": 0.0 * wdb, "bias": 0.0 * wdb, "other": 0.},
        })  # pyformat: disable

    config = ml_collections.ConfigDict()
    config.lr = lrb
    config.lr_mults = [
        ("Dense_0/.*", lr1),
        ("Dense_1/.*", lr2),
    ]
    config.wd = wdb
    config.wd_mults = [
        (".*/kernel", wd1),
        (".*/bias", wd2),
    ]
    mult1 = 1.0
    mult2 = 0.1
    config.schedule = [
        ("Dense_0/.*", dict(decay_type="linear", mult=mult1, linear_end=mult1)),
        ("Dense_[12]/.*", dict(decay_type="linear", mult=mult2)),
        (".*", None),
    ]
    config.optax_name = "scale"
    config.grad_clip_norm = 1.0
    config.optax = ml_collections.ConfigDict()
    g_scale = 0.5
    config.optax.step_size = g_scale

    total_steps = 10
    sched_kw = dict(global_batch_size=1, total_steps=total_steps)
    tx, (sched_fn1,
         sched_fn2) = bv_optax.make(config, params, sched_kw=sched_kw)
    opt_state = tx.init(params)

    # Manually specify schedules for computing expected values.
    frozen_fn = lambda _: jnp.array(0.)
    sched_fns = {
        "Dense_0": {"kernel": sched_fn1, "bias": sched_fn1, "other": sched_fn1},
        "Dense_1": {"kernel": sched_fn2, "bias": sched_fn2, "other": sched_fn2},
        "Dense_2": {"kernel": sched_fn2, "bias": sched_fn2, "other": sched_fn2},
        "Dense_3": {"kernel": frozen_fn, "bias": frozen_fn, "other": frozen_fn},
    }  # pyformat: disable

    grads = jax.tree.map(jnp.ones_like, params)
    gflat, _ = jax.tree.flatten(
        # Don't count frozen params towards gradient norm.
        jax.tree.map(lambda g, sched_fn: {frozen_fn: 0}.get(sched_fn, g),
                     grads, sched_fns))
    l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in gflat]))
    grad_clip_factor = jnp.minimum(1.0, config.grad_clip_norm / l2_g)
    grads_scaled = jax.tree.map(lambda p: grad_clip_factor * p, grads)

    def make_tx(step):
      def get_update(p, g, wd, sched_fn, lr_mult):
        return -sched_fn(step) * (lrb * lr_mult * g_scale * g + p * wd)
      return get_update

    for step in range(total_steps):
      updates, opt_state = tx.update(grads, opt_state, params)
      self.assertEqual(bv_optax.get_count(opt_state), step + 1)
      sched1, sched2 = sched_fn1(step), sched_fn2(step)
      np.testing.assert_almost_equal(sched1, mult1)
      np.testing.assert_almost_equal(sched2,
                                     mult2 * (total_steps - step) / total_steps)
      chex.assert_trees_all_close(
          updates,
          jax.tree.map(
              make_tx(step), params, grads_scaled, wds, sched_fns, lr_mults))

  def test_frozen_no_state(self):
    params = {"small": jnp.zeros([1]), "large": jnp.zeros([1000])}
    config = ml_collections.ConfigDict()
    config.lr = 0.01
    config.schedule = [
        ("small", dict(decay_type="cosine")),
        ("large", None),
    ]
    config.optax_name = "scale_by_adam"

    sched_kw = dict(global_batch_size=1, total_steps=1)
    tx, _ = bv_optax.make(config, params, sched_kw=sched_kw)

    opt_state = tx.init(params)
    adam_state = bv_optax.find_states(opt_state, optax.ScaleByAdamState)
    nbytes = sum(
        jax.tree.flatten(jax.tree.map(lambda x: x.nbytes, adam_state))[0])
    self.assertLess(nbytes, 1_000)

  def test_adafactor(self):
    params = {"Dense_0": {"kernel": jnp.zeros([1024, 1024])}}

    config = ml_collections.ConfigDict()
    config.optax_name = "big_vision.scale_by_adafactor"
    config.lr = 0.01
    config.schedule = dict(decay_type="linear")
    sched_kw = dict(global_batch_size=1, total_steps=1)

    tx, _ = bv_optax.make(config, params, sched_kw=sched_kw)

    opt_state = tx.init(params)
    adafactor_state = bv_optax.find_states(opt_state, optax.FactoredState)
    n_state_params = sum(
        jax.tree.flatten(
            jax.tree.map(lambda x: np.prod(
                x.shape if hasattr(x, "shape") else 0), adafactor_state))[0])
    self.assertEqual(n_state_params, 2 * 1024 + 2)


if __name__ == "__main__":
  absltest.main()
