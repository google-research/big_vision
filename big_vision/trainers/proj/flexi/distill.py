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

"""Distill a teacher model into a FlexiViT student.

Note this file has code that is generic enough to allow using an ensemble
of teachers. This is inherited from `proj/distill/distill.py` and the goal
to only make minimal changes in a fork of that file. However, this feature
does not really make sense for FlexiViT.
"""
# pylint: disable=consider-using-from-import
from functools import partial
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
import big_vision.evaluators.common as eval_common
import big_vision.evaluators.proj.distill.distance as dd
import big_vision.input_pipeline as input_pipeline
import big_vision.optax as bv_optax
import big_vision.trainers.proj.flexi.common as flexi
import big_vision.utils as u
from clu import parameter_overview
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf

from tensorflow.io import gfile

# pylint: disable=logging-fstring-interpolation


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def getfirst(d, *keys):
  """Returns the first of `keys` that's present in mapping `d`."""
  result, found = None, False
  for k in reversed(keys):
    if k in d:
      result, found = d[k], True
  if found:
    return result
  else:
    raise KeyError(f"None of {keys} is in {d.keys()}")


def main(argv):
  del argv
  tf.config.experimental.set_visible_devices([], "GPU")

  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)
    save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules", ["ops_general", "ops_image", "ops_text"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  # See (internal link) for a fun read on what it takes.
  rng = jax.random.PRNGKey(config.get("seed", 0))

  # These functions do more stuff internally, for OSS release we mock them by
  # trivial alternatives in order to minize disruptions in the code.
  xid, wid = -1, -1
  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  write_note("Initializing...")

  batch_size = config.input.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must "
                     f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  # First thing after above sanity checks, so we can log "start" ticks.
  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

  write_note("Initializing train dataset...")
  train_ds, ntrain_img = input_pipeline.training(config.input)

  # Start prefetching already.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch)

  total_steps = u.steps("total", config, ntrain_img, batch_size)
  def get_steps(name, default=ValueError, cfg=config):
    return u.steps(name, cfg, ntrain_img, batch_size, total_steps, default)

  u.chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size,
                  measure=mw.measure, write_note=write_note)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Create student and teacher models
  def get_model_mod(name):  # Used many times.
    mod_name = config[f"{name}_name"]
    return importlib.import_module(f"big_vision.models.{mod_name}")

  write_note("Initializing models...")
  def make_model(name):
    return get_model_mod(name).Model(
        num_classes=config.num_classes, **config.get(name, {}))

  models = {
      "student": make_model("student"),
      **{t: make_model(t) for t in config.teachers}
  }

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  def get_init(model, name):
    @partial(jax.jit, backend="cpu")
    def _init(rng):
      bs = batch_size // jax.device_count()
      img_size = tuple(getfirst(train_ds.element_spec, name, "image").shape[1:])
      no_image = jnp.zeros((bs,) + img_size, jnp.float32)
      params = flax.core.unfreeze(model.init(rng, no_image))["params"]
      return params
    return _init

  rng, *rng_inits = jax.random.split(rng, len(models) + 1)
  with u.chrono.log_timing("z/secs/init"):
    params_cpu = {
        name: get_init(models[name], name=name)(r)
        for name, r in zip(models, rng_inits)}

  if jax.process_index() == 0:
    for name, params in params_cpu.items():
      parameter_overview.log_parameter_overview(params, msg=f"{name} params")
      mw.measure(f"num_params_{name}",
                 sum(p.size for p in jax.tree_leaves(params)))

  write_note(f"Initializing {config.optax_name} optimizer...")
  # For now, we explicitly only optimize the student parameters as there's
  # nothing else to be optimized. If we ever want to add learnable projections
  # or similar for good (we explored but ditched), need to refactor this a bit.
  tx, sched_fns = bv_optax.make(
      config, params_cpu["student"], sched_kw=dict(
          total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))

  # We jit this, such that the arrays are created on the CPU, not device[0].
  opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu["student"])
  sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]

  @jax.named_call
  def loss_fn(student_params, params, data, rngs, **flexi_kw):
    # Note: need to extract and use `student_params` out of `params` because the
    # first argument of `loss_fn` is what's differentiated wrt.
    params["student"] = student_params

    def fwd(name, params):
      return jax.named_call(models[name].apply, name=name)(
          {"params": params}, getfirst(data, name, "image"),
          train=name == "student", rngs=rngs.get(name),
          **(flexi_kw if name == "student" else {})
      )[0]  # logits, unused_outputs
    logits = {name: fwd(name, w) for name, w in params.items()}

    measurements = {}
    for name, lg in logits.items():
      measurements[f"entropy_{name}"] = -jnp.sum(
          jax.nn.log_softmax(lg) * jax.nn.softmax(lg), axis=-1)
      if "labels" in data:
        measurements[f"task_loss_{name}"] = u.softmax_xent(
            logits=lg, labels=data["labels"], reduction=False)

    # NOTE: xent is linear in labels, so for KL, this is actually the same as
    # using a teacher-ensemble in probs-space!
    measurements["distill_loss"] = 0.0
    for name in config.teachers:
      l = dd.dist(logits["student"], logits[name], config.get("distance", "kl"),
                  **config.get("distance_kw", {}))
      measurements[f"distill_loss_{name}"] = l
      measurements["distill_loss"] += l

    outputs = (measurements["distill_loss"], measurements)
    return jax.tree_map(jnp.mean, outputs)

  flexi_argnames = sorted(config.flexi)

  @partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1),
           static_broadcasted_argnums=tuple(range(4, 4 + len(flexi_argnames))))
  def update_fn(params, opt, rng, data, *args):
    """Update step."""

    # Mixup. Note: overwrites the `data` entries (that's intended).
    if config.get("mixup") and config.mixup.p:
      to_mix = {name: data[name]
                for name in ("image", "labels") + tuple(models) if name in data}
      rng, _, to_mix = u.mixup(rng, **config.mixup, **to_mix)
      data = {**data, **to_mix}

    # Get device-specific loss rng.
    rng, *rng_models = jax.random.split(rng, len(models) + 1)
    rngs_models_local = {
        name: {"dropout": jax.random.fold_in(rngi, jax.lax.axis_index("batch"))}
        for name, rngi in zip(models, rng_models)
    }

    w = params["student"]  # Need to explicitly pull out the optimized ones.
    (l, measurements), grads = jax.lax.pmean(
        jax.value_and_grad(loss_fn, has_aux=True)(
            w, params, data, rngs=rngs_models_local,
            **dict(zip(flexi_argnames, args))),
        axis_name="batch")
    updates, opt = tx.update(grads, opt, w)
    w = optax.apply_updates(w, updates)
    params["student"] = w

    # Take some logging measurements
    gs = jax.tree_leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree_leaves(w)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    return params, opt, rng, l, measurements

  # We always load the teachers first, because they NEED to be initialized
  # and since we don't ever modify them, we don't store them in checkpoints.
  for name in config.teachers:
    init_def = config[f"{name}_init"]
    write_note(f"Initializing {name} from {init_def}…")
    params_cpu[name] = get_model_mod(name).load(
        params_cpu[name], init_def, config[name],
        **config.get(f"{name}_load", {}))

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize student from something, e.g. start a fine-tuning job.
  # 4. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(save_ckpt_path):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = fillin(config.resume)
  if resume_ckpt_path:
    write_note("Resume training from checkpoint...")
    # NOTE: we never change the teachers, so only checkpoint student here.
    checkpoint = {
        "params": params_cpu["student"],
        "opt": opt_cpu,
        "chrono": u.chrono.save(),
    }
    checkpoint_tree = jax.tree_structure(checkpoint)
    loaded = u.load_checkpoint(checkpoint_tree, resume_ckpt_path)
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    checkpoint = jax.tree_map(u.recover_dtype, loaded)
    params_cpu["student"], opt_cpu = checkpoint["params"], checkpoint["opt"]
    u.chrono.load(checkpoint["chrono"])
  elif config.get("student_init"):
    write_note(f"Initialize student from {config.student_init}...")
    params_cpu["student"] = get_model_mod("student").load(
        params_cpu["student"], config.student_init, config.get("student"),
        **config.get("student_load", {}))
    if jax.process_index() == 0:
      parameter_overview.log_parameter_overview(
          params_cpu["student"], msg="restored (student) params")

  write_note("Kicking off misc stuff...")
  first_step = bv_optax.get_count(opt_cpu)
  u.chrono.inform(first_step=first_step)
  prof = None  # Keeps track of start/stop of profiler state.

  write_note(f"Replicating...\n{u.chrono.note}")
  params_repl = flax.jax_utils.replicate(params_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  # Initializing evaluators later when they are first needed, so we can see
  # issues with training faster.
  evaluators = None

  # Define predict functions that the evaluators can use:
  def predict_fn(params, *, name, **kw):
    image = kw.pop(name, kw.pop("image", None))
    # Ugly API compatibility necessity:
    for k in ("student", *config.teachers):
      kw.pop(k, 0)
    return models[name].apply({"params": params[name]}, image, **kw)

  # 1. One for each variant of the student
  student_pfns = flexi.mkpredictfns(partial(predict_fn, name="student"),
                                    config.flexi, "student_{x}")
  # 2. One per teacher model
  teacher_pfns = {
      name: partial(predict_fn, name=name)
      for name in config.teachers
  }
  # 3. One for each (student-variant, teacher) pair, eg for distance eval.
  combined_pfns = {
      f"{sn}_{tn}": lambda *a, sfn=sfn, tfn=tfn, **kw: (sfn(*a, **kw), tfn(*a, **kw))  # pylint: disable=line-too-long
      for sn, sfn in student_pfns.items()
      for tn, tfn in teacher_pfns.items()
  }

  predict_fns = {**student_pfns, **teacher_pfns, **combined_pfns}

  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  ckpt_writer = None

  write_note(f"First step compilations...\n{u.chrono.note}")

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)

    np_rng = flexi.mkrng(xid, wid, step)
    flexi_args = [
        flexi.choice(config.flexi[n].v, config.flexi[n].p, np_rng)
        for n in flexi_argnames
    ]

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with u.chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
        params_repl, opt_repl, rngs_loop, loss_value, measurements = update_fn(
            params_repl, opt_repl, rngs_loop, batch, *flexi_args)

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
        or u.chrono.warmup and jax.process_index() == 0):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}", sched_fn_cpu(step - 1))
      l = mw.measure("training_loss", loss_value[0])
      for name, value in measurements.items():
        mw.measure(name, value[0])
      u.chrono.tick(step)
      if not np.isfinite(l):
        raise RuntimeError(f"The loss became nan or inf somewhere within steps "
                           f"[{step - get_steps('log_training')}, {step}]")

    # Checkpoint saving
    if (save_ckpt_path and
        (u.itstime(step, get_steps("ckpt", None), total_steps, host=0) or
         u.itstime(step, get_steps("keep_ckpt", None), total_steps, host=0))):
      u.chrono.pause(wait_for=(params_repl["student"], opt_repl))
      u.checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see (internal link)). Also, takes device 0's params only.
      params_cpu["student"], opt_cpu = jax.tree_map(
          lambda x: np.array(x[0]), (params_repl["student"], opt_repl))

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if u.itstime(step, get_steps("keep_ckpt", None), total_steps):
        copy_step = step

      ckpt = {"params": params_cpu["student"],
              "opt": opt_cpu,
              "chrono": u.chrono.save()}
      ckpt_writer = pool.apply_async(
          u.save_checkpoint, (ckpt, save_ckpt_path, copy_step))
      u.chrono.resume()

    if evaluators is None:
      evaluators = eval_common.from_config(
          config, predict_fns,
          lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
          lambda key, cfg: get_steps(key, default=None, cfg=cfg),
      )
    for (name, evaluator, log_steps, prefix) in evaluators:
      if u.itstime(step, log_steps, total_steps):
        u.chrono.pause(wait_for=params_repl)
        u.chrono.tick(step)  # Record things like epoch number, core hours etc.
        write_note(f"{name} evaluation...\n{u.chrono.note}")
        with u.chrono.log_timing(f"z/secs/eval/{name}"):
          for key, value in evaluator.run(params_repl):
            mw.measure(f"{prefix}{key}", value)
        u.chrono.resume()
    mw.step_end()

  # Always give a chance to stop the profiler, no matter how things ended.
  # TODO: can we also do this when dying of an exception like OOM?
  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  # Last note needs to happen before the pool's closed =)
  write_note(f"Done!\n{u.chrono.note}")

  pool.close()
  pool.join()
  mw.close()

  # Make sure all hosts stay up until the end of main.
  u.sync()

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)
