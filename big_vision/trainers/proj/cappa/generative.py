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

"""Training loop for CapPa (https://arxiv.org/abs/2306.07915)."""
# pylint: disable=consider-using-from-import
import functools
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
import big_vision.datasets.core as ds_core
import big_vision.evaluators.common as eval_common
import big_vision.input_pipeline as input_pipeline
import big_vision.optax as bv_optax
import big_vision.sharding as bv_sharding
import big_vision.trainers.proj.cappa.predict_fns as predict_fns
import big_vision.utils as u
from clu import parameter_overview
import flax
import flax.linen as nn
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serial
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
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
jax.config.update("jax_transfer_guard", "disallow")
# Fixes design flaw in jax.random that may cause unnecessary d2d comms.
jax.config.update("jax_threefry_partitionable", True)


def main(argv):
  del argv

  try:
    jax.distributed.initialize()
  except ValueError as e:
    logging.warning('Could not initialize distributed environment: %s', e)

  # Make sure TF does not touch GPUs.
  tf.config.set_visible_devices([], "GPU")

  # Consistent device order is important to ensure correctness of various train
  # loop components, such as input pipeline, update step, evaluators. We use
  # jax utils to infer device order that will be used throughout the program.
  config = flags.FLAGS.config
  num_sharded_replicas = config.get("num_sharded_replicas", 1)
  assert jax.device_count() % num_sharded_replicas == 0, (
      num_sharded_replicas, jax.device_count())
  devices = mesh_utils.create_device_mesh(
      (num_sharded_replicas, jax.device_count() // num_sharded_replicas))
  devices_flat = devices.reshape(-1)

################################################################################
#                                                                              #
#                                Set up logging                                #
#                                                                              #
################################################################################

  # Set up work directory and print welcome message.
  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)
    save_ckpt_path = os.path.join(workdir, "checkpoint.bv")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules", ["ops_general", "ops_image", "ops_text"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # Setup up logging and experiment manager.
  xid, wid = -1, -1
  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

################################################################################
#                                                                              #
#                                Input Pipeline                                #
#                                                                              #
################################################################################

  write_note("Initializing train dataset...")
  batch_size = config.input.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must "
                     f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  train_ds, ntrain_img = input_pipeline.training(config.input)

  total_steps = u.steps("total", config, ntrain_img, batch_size)
  def get_steps(name, default=ValueError, cfg=config):
    return u.steps(name, cfg, ntrain_img, batch_size, total_steps, default)

  u.chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size,
                  measure=mw.measure, write_note=write_note)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Start input pipeline as early as possible.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_global(train_ds, devices_flat, n_prefetch)

  # For mixed data, add per-dataset epoch and examples seen measurements.
  if isinstance(config.input.data.get("name"), str):
    measure_per_dataset_times = lambda step: None  # No-op
  else:
    nexamples = {
        name: ds_core.get(**config.input[name].data).total_examples
        for name in config.input.data
    }
    def measure_per_dataset_times(step):
      total = sum(config.input.data.values())
      for name, w in config.input.data.items():
        w = w / total
        mw.measure(f"examples_seen_{name}", u.chrono.accum_examples_seen * w)
        mw.measure(f"epoch_{name}", step * batch_size * w / nexamples[name])

################################################################################
#                                                                              #
#                           Create Model & Optimizer                           #
#                                                                              #
################################################################################

  write_note(f"Initializing {config.model_name} model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model = model_mod.Model(**config.get("model", {}))

  def init(rng):
    bs = batch_size // jax.device_count()
    img_shape = (bs,) + tuple(train_ds.element_spec["image"].shape[1:])
    txt_shape = (bs,) + tuple(train_ds.element_spec["labels"].shape[1:])
    dummy_img = jnp.zeros(img_shape, jnp.float32)
    dummy_txt = jnp.zeros(txt_shape, jnp.int64)
    variables = model.init(rng, dummy_img, dummy_txt)
    return flax.core.unfreeze(variables["params"])

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  # See (internal link) for a fun read on what it takes.
  rng = jax.random.PRNGKey(u.put_cpu(config.get("seed", 0)))

  rng, rng_init = jax.random.split(rng)
  with u.chrono.log_timing("z/secs/init"):
    write_note("Inferring parameter shapes...")
    params_shape = jax.eval_shape(init, rng_init)

    write_note("Inferring optimizer state shapes...")
    tx, sched_fns = bv_optax.make(config, params_shape, sched_kw=dict(
        total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))
    opt_shape = jax.eval_shape(tx.init, params_shape)
    # We jit this, such that the arrays are created on the CPU, not device[0].
    sched_fns_cpu = [u.jit_cpu()(sched_fn) for sched_fn in sched_fns]

    num_params = sum(np.prod(p.shape) for p in jax.tree_leaves(params_shape))
    mw.measure("num_params", num_params)

################################################################################
#                                                                              #
#                               Shard & Transfer                               #
#                                                                              #
################################################################################

  # Using a 2D mesh where the model/optimizer parameters are replicated along
  # the "replica" axis and sharded along the "fsdp" axis if the sharding is set
  # to "fully_sharded" for the model/optimizer.
  write_note("Creating device mesh...")
  mesh = jax.sharding.Mesh(devices, ("replica", "fsdp"))
  repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  write_note("Inferring shardings...")
  params_sharding = bv_sharding.infer_sharding(
      params_shape, mesh, axis_name="fsdp",
      # TODO: implement scan for parameter sharding.
      strategy=config.get("param_sharding", "replicated"),
      extra_strategy_args=config.get("param_sharding_args", {}))
  opt_sharding = bv_sharding.infer_sharding(
      opt_shape, mesh, axis_name="fsdp",
      strategy=config.get("optim_sharding", "replicated"),
      extra_strategy_args=config.get("optim_sharding_args", {}))

  write_note("Transferring train_state to devices...")
  # RNG is always replicated
  rng_init = u.reshard(rng_init, repl_sharding)

  # Parameters and the optimizer are now global (distributed) jax arrays.
  params = jax.jit(init, out_shardings=params_sharding)(rng_init)
  opt = jax.jit(tx.init, out_shardings=opt_sharding)(params)

  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)
  del rng  # not used anymore, so delete it.

  # At this point we have everything we need to form a train state. It contains
  # all the parameters that are passed and updated by the main training step.
  train_state_sharding = {
      "params": params_sharding, "opt": opt_sharding, "rng": repl_sharding}
  train_state = {
      "params": params, "opt": opt, "rng": rng_loop}
  del params, opt, rng_loop  # Delete to avoid memory leak or accidental reuse.

  write_note("Logging parameter overview...")
  parameter_overview.log_parameter_overview(
      train_state["params"], msg="Init params",
      include_stats="global", jax_logging_process=0)

################################################################################
#                                                                              #
#                                 Update Step                                  #
#                                                                              #
################################################################################

  @functools.partial(
      jax.jit,
      donate_argnums=(0,),
      out_shardings=(train_state_sharding, repl_sharding))
  def update_fn(train_state, batch):
    """Update step."""

    images, labels, label_masks = (
        batch["image"], batch["labels"], batch.get("label_masks"))

    # Get device-specific loss rng.
    rng = train_state["rng"]
    rng, rng_model = jax.random.split(rng, 2)

    def loss_fn(params):
      logits = model.apply(
          {"params": params}, images, labels,
          train=True, rngs={"dropout": rng_model})

      weights = jnp.where(labels != config.get("pad_token", 0), 1.0, 0.0)
      if label_masks is not None:
        weights = weights * label_masks

      loss = u.weighted_softmax_xent(
          logits=logits, labels=labels,
          weights=weights, label_smoothing=config.get("label_smoothing", 0.0),
          reduction=True, normalize=True)

      return loss

    params, opt = train_state["params"], train_state["opt"]
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    measurements = {"training_loss": loss}
    gs = jax.tree_leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.sum(g * g) for g in gs]))
    ps = jax.tree_leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.sum(p * p) for p in ps]))
    us = jax.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.sum(u * u) for u in us]))

    return {"params": params, "opt": opt, "rng": rng}, measurements

################################################################################
#                                                                              #
#                               Load Checkpoint                                #
#                                                                              #
################################################################################

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from something, e,g, start a fine-tuning job.
  # 4. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(f"{save_ckpt_path}-LAST"):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = fillin(config.resume)

  ckpt_mngr = None
  if save_ckpt_path or resume_ckpt_path:
    ckpt_mngr = array_serial.GlobalAsyncCheckpointManager()

  if resume_ckpt_path:
    write_note(f"Resuming training from checkpoint {resume_ckpt_path}...")
    jax.tree_map(lambda x: x.delete(), train_state)
    del train_state
    shardings = {
        **train_state_sharding,
        "chrono": jax.tree_map(lambda _: repl_sharding,
                               u.chrono.save()),
    }
    loaded = u.load_checkpoint_ts(
        resume_ckpt_path, tree=shardings, shardings=shardings)
    train_state = {key: loaded[key] for key in train_state_sharding.keys()}

    u.chrono.load(jax.device_get(loaded["chrono"]))
    del loaded
  elif config.get("model_init"):
    write_note(f"Initialize model from {config.model_init}...")
    train_state["params"] = model_mod.load(
        train_state["params"], config.model_init, config.get("model"),
        **config.get("model_load", {}))

    # load has the freedom to return params not correctly sharded. Think of for
    # example ViT resampling position embedings on CPU as numpy arrays.
    train_state["params"] = u.reshard(
        train_state["params"], train_state_sharding["params"])

    parameter_overview.log_parameter_overview(
        train_state["params"], msg="restored params",
        include_stats="global", jax_logging_process=0)

################################################################################
#                                                                              #
#                                 Setup Evals                                  #
#                                                                              #
################################################################################

  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config,
        predict_fns.get_predict_fns(model),
        lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        devices_flat,
    )

  # At this point we need to know the current step to see whether to run evals.
  write_note("Inferring the first step number...")
  first_step_device = bv_optax.get_count(train_state["opt"], jittable=True)
  first_step = int(jax.device_get(first_step_device))
  u.chrono.inform(first_step=first_step)

  # Note that training can be pre-empted during the final evaluation (i.e.
  # just after the final checkpoint has been written to disc), in which case we
  # want to run the evals.
  if first_step in (total_steps, 0):
    write_note("Running initial or final evals...")
    mw.step_start(first_step)
    for (name, evaluator, _, prefix) in evaluators():
      if config.evals[name].get("skip_first") and first_step != total_steps:
        continue
      write_note(f"{name} evaluation...\n{u.chrono.note}")
      with u.chrono.log_timing(f"z/secs/eval/{name}"):
        with mesh, nn.logical_axis_rules([("act_batch", ("replica", "fsdp"))]):  # pytype: disable=wrong-arg-types
          for key, value in evaluator.run(train_state):
            mw.measure(f"{prefix}{key}", value)

################################################################################
#                                                                              #
#                                  Train Loop                                  #
#                                                                              #
################################################################################

  prof = None  # Keeps track of start/stop of profiler state.

  write_note("Starting training loop, compiling the first step...")
  for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with u.chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
        with mesh, nn.logical_axis_rules([("act_batch", ("replica", "fsdp"))]):  # pytype: disable=wrong-arg-types
          train_state, measurements = update_fn(train_state, batch)

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
        or u.chrono.warmup and jax.process_index() == 0):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}",
                   sched_fn_cpu(u.put_cpu(step - 1)))
      measurements = jax.device_get(measurements)
      for name, value in measurements.items():
        mw.measure(name, value)
      u.chrono.tick(step)
      measure_per_dataset_times(step)

      if not np.isfinite(measurements["training_loss"]):
        raise RuntimeError(f"The loss became nan or inf somewhere within steps "
                           f"[{step - get_steps('log_training')}, {step}]")

    # Checkpoint saving
    keep_ckpt_steps = get_steps("keep_ckpt", None) or total_steps
    if save_ckpt_path and (
        (keep := u.itstime(step, keep_ckpt_steps, total_steps, first=False))
        or u.itstime(step, get_steps("ckpt", None), total_steps, first=True)
    ):
      u.chrono.pause(wait_for=train_state)

      # Copy because we add extra stuff to the checkpoint.
      ckpt = {**train_state}

      # To save chrono state correctly and safely in a multihost setup, we
      # broadcast the state to all hosts and convert it to a global array.
      with jax.transfer_guard("allow"):
        chrono_ckpt = multihost_utils.broadcast_one_to_all(u.chrono.save())
      chrono_shardings = jax.tree_map(lambda _: repl_sharding, chrono_ckpt)
      ckpt = ckpt | {"chrono": u.reshard(chrono_ckpt, chrono_shardings)}

      u.save_checkpoint_ts(ckpt_mngr, ckpt, save_ckpt_path, step, keep)
      u.chrono.resume()

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
        u.chrono.pause(wait_for=train_state)
        u.chrono.tick(step)  # Record things like epoch number, core hours etc.
        write_note(f"{name} evaluation...\n{u.chrono.note}")
        with u.chrono.log_timing(f"z/secs/eval/{name}"):
          with mesh, nn.logical_axis_rules(
              [("act_batch", ("replica", "fsdp"))]):  # pytype: disable=wrong-arg-types
            for key, value in evaluator.run(train_state):
              mw.measure(f"{prefix}{key}", jax.device_get(value))
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

  if ckpt_mngr:
    ckpt_mngr.wait_until_finished()

  # Make sure all hosts stay up until the end of main.
  u.sync()

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)
