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

"""Train loop for training the stage-II model."""
# pylint: disable=consider-using-from-import
import functools
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
from big_vision import input_pipeline
import big_vision.datasets.core as ds_core
import big_vision.evaluators.common as eval_common
import big_vision.models.proj.uvim.decode as decode
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

import tensorflow.io.gfile as gfile


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


FLAGS = flags.FLAGS
ONE_HOT_AXIS = -2
partial = functools.partial


def get_model(config):
  mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model = mod.Model(**config.model)
  return model, mod


def setup_task(config):
  """Get functions and params to encode and decode labels as token sequences."""
  config = config.oracle

  # Define task input and predict functions.
  task_module = importlib.import_module(f"big_vision.trainers.{config.task}")
  input_fn = partial(task_module.input_pp, config=config)
  predict_outputs_fn = partial(task_module.predict_outputs, config=config)

  oracle, mod = get_model(config)
  if config.get("model_init", None):
    params, state = mod.load(None, config.model_init)
    params = {"params": params, "state": state}
  else:
    params = {}

  def encode_labels(params, batch):
    inputs = input_fn(batch)
    code = oracle.apply(params, **inputs, method=oracle.encode)[1]["code"]
    return code + 1  # To avoid padding symbol.

  def decode_labels(params, code, batch, **kwargs):
    code = code - 1
    inputs = input_fn(batch)
    inputs["x"] = code
    logits, _ = oracle.apply(
        params, **inputs, discrete_input=True, **kwargs, method=oracle.decode)
    return logits

  return encode_labels, decode_labels, predict_outputs_fn, params


def main(argv):
  del argv

  config = FLAGS.config
  workdir = FLAGS.workdir
  logging.info("\u001b[33mHello from process %i holding %i/%i devices and "
               "writing to workdir %s.\u001b[0m", jax.process_index(),
               jax.local_device_count(), jax.device_count(), workdir)

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)
    save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules",
                      ["ops_general", "ops_image", "proj.uvim.pp_ops"]):
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
  chrono = u.Chrono()

  write_note("Initializing train dataset...")
  train_data = ds_core.get(**config.input.data)
  train_ds = input_pipeline.make_for_train(
      data=train_data.get_tfdata(ordered=False),
      batch_size=batch_size,
      preprocess_fn=pp_builder.get_preprocess_fn(config.input.get("pp")),
      shuffle_buffer_size=config.input.get("shuffle_buffer_size"),
      cache_raw=config.input.get("cache_raw", False),
      filter_fn=config.input.get("filter_fn"),
  )

  # Start prefetching already.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch)
  ntrain_img = train_data.total_examples

  def get_steps(name, default=ValueError):  # partial doesn't work well here.
    return u.steps(name, config, ntrain_img, batch_size, default)
  total_steps = get_steps("total")

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  write_note(f"Initializing {config.model_name} model...")
  model, model_mod = get_model(config)

  encode_labels, decode_labels, predict_outputs_fn, task_params = (
      setup_task(config))

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend="cpu")
  def init(rng):
    batch = jax.tree_map(
        lambda x: jnp.zeros(x.shape, x.dtype.as_numpy_dtype),
        train_ds.element_spec)
    images = batch["image"]
    labels = encode_labels(task_params, batch)
    variables = model.init(rng, images, labels)
    params = flax.core.unfreeze(variables["params"])
    return params

  rng, init_rng = jax.random.split(rng)
  params_cpu = init(init_rng)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_leaves(params_cpu))
    parameter_overview.log_parameter_overview(params_cpu, msg="init params")
    mw.measure("num_params", num_params)

  write_note(f"Initializing {config.optax_name} optimizer...")
  tx, sched_fns = bv_optax.make(config, params_cpu, sched_kw=dict(
      total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))

  # We jit this, such that the arrays are created on the CPU, not device[0].
  opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
  sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]

  @partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
  def update_fn(params, opt, batch, update_rng, task_params):
    """Update step."""
    images = batch["image"]
    labels = encode_labels(task_params, batch)

    measurements = {}

    rng, new_rng = jax.random.split(update_rng)
    # bind the rng key to the device id (which is unique across hosts)
    rng_local = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

    def loss_fn(params, images, labels):
      logits = model.apply({"params": params}, images, labels, train=True,
                           rngs={"dropout": rng_local})
      loss = u.weighted_softmax_xent(
          logits=logits, labels=labels,
          reduction=True, normalize=True)
      return loss

    l, grads = jax.value_and_grad(loss_fn)(params, images, labels)
    l, grads = jax.lax.pmean((l, grads), axis_name="batch")
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    gs = jax.tree_leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree_leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    return params, opt, l, new_rng, measurements

  # Define evaluators.
  def validation_fn(params, batch):
    """Compute per-example metrics."""
    params, task_params = params["params"], params["task_params"]
    images = batch["image"]
    labels = encode_labels(task_params, batch)
    logits = model.apply({"params": params}, images, labels, train=False)
    loss = u.weighted_softmax_xent(
        logits=logits, labels=labels,
        reduction=False, normalize=True)
    losses = {"loss": loss}
    return jax.tree_map(
        lambda x: jnp.mean(x, axis=tuple(range(1, x.ndim))),
        losses)

  def predict_fn(params, batch, seed=0, temperature=1e-7, **extra):
    params, task_params = params["params"], params["task_params"]

    # Derive a rng key from the inputs so that all batches use different keys.
    if "image/id" in batch:
      key = batch["image/id"]
    else:
      key = batch["image"].sum(axis=[1, 2, 3]).astype(jnp.int32)
    local_rng = jax.lax.scan(
        lambda k, x: (jax.random.fold_in(k, x), None),
        jax.random.PRNGKey(seed),
        key,
    )[0]

    images = batch["image"]
    batch_size = images.shape[0]
    prompts = jnp.zeros((batch_size, config.model.seq_len), dtype=jnp.int32)
    seqs, _, _ = decode.temperature_sampling(
        params={"params": params}, model=model, seed=local_rng,
        inputs=images, prompts=prompts,
        num_samples=1, eos_token=-1, prefill=False,
        temperature=temperature)
    seqs = jnp.squeeze(seqs, 1)
    logits = decode_labels(task_params, seqs, batch)
    return predict_outputs_fn(logits, **extra)

  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config, {"predict": predict_fn, "validation": validation_fn},
        lambda s: write_note(f"Initializing evaluator: {s}...\n{chrono.note}")
    )

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from something, e,g, start a fine-tuning job.
  # 4. Initialize part of the model from something, eg. only encoder or decoder.
  # 5. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(save_ckpt_path):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = fillin(config.resume)
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
    params_cpu = model_mod.load(
        params_cpu, config.model_init, config.model,
        **config.get("model_load", {}))
    if jax.process_index() == 0:
      parameter_overview.log_parameter_overview(
          params_cpu, msg="restored params")

  write_note("Kicking off misc stuff...")
  first_step = bv_optax.get_count(opt_cpu)
  chrono.inform(first_step, total_steps, batch_size, ntrain_img / batch_size)
  prof = None  # Keeps track of start/stop of profiler state.

  write_note(f"Replicating...\n{chrono.note}")
  params_repl = flax.jax_utils.replicate(params_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)
  task_params = flax.jax_utils.replicate(task_params)
  update_rngs = flax.jax_utils.replicate(rng)

  ckpt_writer = None

  write_note(f"First step compilations...\n{chrono.note}")
  error = None  # For exiting with an error after cleanup. Avoids indentation.

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      params_repl, opt_repl, loss_value, update_rngs, measurements = (
          update_fn(
              params_repl,
              opt_repl,
              batch,
              update_rng=update_rngs,
              task_params=task_params))

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
        or chrono.warmup and jax.process_index() == 0):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}", sched_fn_cpu(step - 1))
      l = mw.measure("training_loss", loss_value[0])
      for name, value in measurements.items():
        mw.measure(name, value[0])
      chrono.tick(step, mw.measure, write_note)
      if not np.isfinite(l):
        error = (f"The loss became nan or inf somewhere within steps "
                 f"[{step - get_steps('log_training')}, {step}]")
        break

    # Checkpoint saving
    if (save_ckpt_path and
        (u.itstime(step, get_steps("ckpt", None), total_steps, host=0) or
         u.itstime(step, get_steps("keep_ckpt", None), total_steps, host=0))):
      chrono.pause(wait_for=(params_repl, opt_repl))
      u.checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see (internal link)). Also, takes device 0's params only.
      opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)
      params_cpu = jax.tree_map(lambda x: np.array(x[0]), params_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if u.itstime(step, get_steps("keep_ckpt", None), total_steps):
        copy_step = step

      ckpt = {"params": params_cpu, "opt": opt_cpu, "chrono": chrono.save()}
      ckpt_writer = pool.apply_async(
          u.save_checkpoint, (ckpt, save_ckpt_path, copy_step))
      chrono.resume()

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=log_steps < total_steps,
                   last=False):
        chrono.pause(wait_for=(params_repl, task_params))
        write_note(f"{name} evaluation...\n{chrono.note}")
        for key, value in evaluator.run(
            {"params": params_repl, "task_params": task_params}):
          mw.measure(f"{prefix}{key}", value)
        chrono.resume()
    mw.step_end()

  # Always give a chance to stop the profiler, no matter how things ended.
  # TODO: can we also do this when dying of an exception like OOM?
  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  # Run final evalution, also used for eval only jobs (when total_steps == 0).
  for (name, evaluator, _, prefix) in evaluators():
    write_note(f"{name} evaluation...\n{chrono.note}")
    for key, value in evaluator.run(
        {"params": params_repl, "task_params": task_params}):
      mw.measure(f"{prefix}{key}", value)

  # Last note needs to happen before the pool's closed =)
  if not error:
    write_note(f"Done!\n{chrono.note}")
  else:
    write_note(f"Failed!\n{error}\n{chrono.note}")

  pool.close()
  pool.join()
  mw.close()

  # Make sure all hosts stay up until the end of main.
  u.sync()

  # Before cleanup, as cleanup should only run for successful jobs.
  if error is not None:
    raise RuntimeError(error)

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)
