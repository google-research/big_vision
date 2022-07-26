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

"""Contrastive training loop.

For models Like
- LiT (https://arxiv.org/abs/2111.07991)
- CLIP (https://arxiv.org/abs/2103.00020)
"""

from functools import partial
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
from big_vision import input_pipeline
import big_vision.evaluators.common as eval_common
import big_vision.optax as bv_optax
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
from clu import parameter_overview
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax

from tensorflow.io import gfile


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def all_gather(z):
  """All gather and flatten first two dims."""
  gather_flat = lambda x: jnp.concatenate(jax.lax.all_gather(x, "batch"), 0)
  return jax.tree_map(gather_flat, z)


def main(argv):
  del argv

  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info(  # pylint: disable=logging-fstring-interpolation
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  assert not config.get("grad_accum_steps"), "Grad-acc not supported anymore."

  save_ckpt_path = None
  if workdir and config.get("ckpt_steps"):
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
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  write_note("Initializing...")

  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get("keep_ckpt_steps"):
    assert config.get("ckpt_steps"), "Specify `ckpt_steps`."
    assert config.keep_ckpt_steps % config.ckpt_steps == 0, (
        f"`keep_ckpt_steps` ({config.ckpt_steps}) should be"
        f"divisible by `ckpt_steps ({config.ckpt_steps}).`")

  batch_size = config.batch_size
  batch_size_eval = config.get("batch_size_eval", batch_size)
  if (batch_size % jax.device_count() != 0 or
      batch_size_eval % jax.device_count() != 0):
    raise ValueError(f"Batch sizes ({batch_size} and {batch_size_eval}) must "
                     f"be divisible by device number ({jax.device_count()})")

  local_batch_size = batch_size // jax.process_count()
  info(
      "Global batch size %d on %d hosts results in %d local batch size. "
      "With %d dev per host (%d dev total), that's a %d per-device batch size.",
      batch_size, jax.process_count(), local_batch_size,
      jax.local_device_count(), jax.device_count(),
      local_batch_size // jax.local_device_count())

  write_note("Initializing training pipeline...")
  train_ds = input_pipeline.make_for_train(
      dataset=config.dataset,
      split=config.train_split,
      batch_size=config.batch_size,
      preprocess_fn=pp_builder.get_preprocess_fn(config.pp_train),
      shuffle_buffer_size=config.get("shuffle_buffer_size"),
      cache_raw=config.get("cache_raw", False),
      data_dir=config.get("dataset_dir"),
      filter_fn=config.get("filter_fn"),
  )

  # Start prefetching already.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch)
  # We always pad to local_batch_size_eval even when less would be enough in
  # order to minimize memory fragmentation.

  ntrain_img = input_pipeline.get_num_examples(
      config.dataset, config.train_split, data_dir=config.get("dataset_dir"))
  steps_per_epoch = ntrain_img / batch_size

  if config.get("total_epochs"):
    total_steps = int(config.total_epochs * steps_per_epoch)
    assert not config.get("total_steps"), "Set only one of total_(epochs|steps)"
  else:
    total_steps = config.total_steps

  info(
      "Running for %d steps, that means %f epochs and %f steps per epoch",
      total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch)
  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)
  chrono = u.Chrono()

  write_note(f"Initializing {config.model_name} model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model = model_mod.Model(**config.get("model", {}))

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend="cpu")
  def init(rng):
    image_size = tuple(train_ds.element_spec["image"].shape[1:])
    no_image = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    text_size = tuple(train_ds.element_spec["labels"].shape[1:])
    no_text = jnp.zeros((local_batch_size,) + text_size, jnp.int32)
    params = flax.core.unfreeze(model.init(rng, no_image, no_text))["params"]
    return params

  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_leaves(params_cpu))
    parameter_overview.log_parameter_overview(params_cpu)
    mw.measure("num_params", num_params)

  tx, sched_fns = bv_optax.make(
      config,
      params_cpu,
      sched_kw=dict(
          global_batch_size=batch_size,
          total_steps=total_steps,
          steps_per_epoch=steps_per_epoch))
  opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
  sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]

  @partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
  def update_fn(params, opt, batch, rng):
    """Update step."""
    assert "mixup" not in config, "We still have to figure out mixup."

    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))

    def loss_fn(params, images, labels):
      zimg, ztxt, extras = model.apply(
          {"params": params}, images, labels,
          train=True, rngs={"dropout": rng_model_local})

      # Gather representations across cores for larger batch size for loss.
      if config.get("loss_use_global_batch", False):
        zimg, ztxt = all_gather((zimg, ztxt))

      l, l_extras = u.bidirectional_contrastive_loss(
          zimg, ztxt, extras["t"], reduction=True)
      return l, {
          "t": extras["t"],
          "t/parameter": extras["t/parameter"],
          "nimg": jnp.mean(extras["img/norm"]),
          "ntxt": jnp.mean(extras["txt/norm"]),
          **l_extras,
      }

    (l, measurements), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(params, batch["image"], batch["labels"])
    l, measurements, grads = jax.lax.pmean((l, measurements, grads),
                                           axis_name="batch")
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    gs = jax.tree_leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree_leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    return params, opt, l, rng, measurements

  # We require hashable function reference for evaluator.
  def predict_fn(params, image=None, text=None, **unused_kwargs):
    del unused_kwargs  # `unused_kwargs` is to be compatible with few-shot
    zimg, ztxt, out = model.apply({"params": params}, image, text)
    return zimg, ztxt, out

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from something, e,g, start a fine-tuning job.
  # 4. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(save_ckpt_path):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = config.resume.format(wid=xm_wu.id)
  if resume_ckpt_path:
    write_note("Resume training from checkpoint...")
    checkpoint = {
        "params": params_cpu,
        "opt": opt_cpu,
        "chrono": chrono.save(),
    }
    checkpoint_tree = jax.tree_structure(checkpoint)
    loaded = u.load_checkpoint(checkpoint_tree, resume_ckpt_path)
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    checkpoint = jax.tree_map(u.recover_dtype, loaded)
    params_cpu, opt_cpu = checkpoint["params"], checkpoint["opt"]
    chrono.load(checkpoint["chrono"])
  elif config.get("model_init"):
    write_note(f"Initialize model from {config.model_init}...")
    params_cpu = model_mod.load(params_cpu, config.model_init,
                                config.get("model"),
                                **config.get("model_load", {}))
    if jax.process_index() == 0:
      info("Restored parameter overview:")
      parameter_overview.log_parameter_overview(params_cpu)

  write_note("Kicking off misc stuff...")
  first_step = bv_optax.get_count(opt_cpu)
  chrono.inform(first_step, total_steps, batch_size, steps_per_epoch)
  prof = None  # Keeps track of start/stop of profiler state.

  write_note(f"Replicating...\n{chrono.note}")
  params_repl = flax.jax_utils.replicate(params_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  # Initialise evaluators later when they are first needed, so we can see
  # issues with training faster.
  evaluators = None

  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  ckpt_writer = None

  write_note(f"First step compilations...\n{chrono.note}")
  error = None  # For exiting with an error after cleanup. Avoids indentation.

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, train_batch in zip(
      range(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      params_repl, opt_repl, loss_value, rngs_loop, extra_measurements = (
          update_fn(params_repl, opt_repl, train_batch, rng=rngs_loop))

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, config.log_training_steps)

    # Checkpoint saving
    if (save_ckpt_path and
        u.itstime(step, config.get("ckpt_steps"), total_steps, host=0)):
      chrono.pause()
      u.checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see (internal link)). Also, takes device 0's params only.
      opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)
      params_cpu = jax.tree_map(lambda x: np.array(x[0]), params_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if u.itstime(step, config.get("keep_ckpt_steps"), total_steps):
        copy_step = step

      # Checkpoint should be a nested dictionary or FLAX datataclasses from
      # `flax.struct`. Both can be present in a checkpoint.
      checkpoint = {
          "params": params_cpu,
          "opt": opt_cpu,
          "chrono": chrono.save(),
      }
      ckpt_writer = pool.apply_async(
          u.save_checkpoint, (checkpoint, save_ckpt_path, copy_step))
      chrono.resume()

    # Report training progress
    if u.itstime(step, config.log_training_steps, total_steps, host=0):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}", sched_fn_cpu(step - 1))
      l = mw.measure("training_loss", loss_value[0])
      for name, value in extra_measurements.items():
        mw.measure(name, value[0])
      chrono.tick(step, mw.measure, write_note)
      if not np.isfinite(l):
        error = (f"The loss became nan or inf somewhere within steps "
                 f"[{step - config.log_training_steps}, {step}]")
        break

    chrono.pause()
    if evaluators is None:
      evaluators = eval_common.from_config(
          config, {"predict": predict_fn},
          lambda s: write_note(f"Initializing evaluator: {s}...\n{chrono.note}")
      )
    for (name, evaluator, freq, prefix) in evaluators:
      if u.itstime(step, freq, total_steps):
        write_note(f"{name} evaluation...\n{chrono.note}")
        for key, value in evaluator.run(params_repl):
          mw.measure(f"{prefix}{key}", value)
    chrono.resume()
    mw.step_end()

  # Always give a chance to stop the profiler, no matter how things ended.
  # TODO: can we also do this when dying of an exception like OOM?
  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  # Last note needs to happen before the pool's closed =)
  if not error:
    write_note(f"Done!\n{chrono.note}")
  else:
    write_note(f"Failed!\n{error}\n{chrono.note}")

  pool.close()
  pool.join()
  mw.close()

  # Make sure all hosts stay up until the end of main.
  u.sync_all_hosts()

  # Before cleanup, as cleanup should only run for successful jobs.
  if error is not None:
    raise RuntimeError(error)

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, logging.info)


if __name__ == "__main__":
  app.run(main)
