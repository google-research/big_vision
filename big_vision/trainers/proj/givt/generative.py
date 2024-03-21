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

"""Training loop for GIVT-style autoregressive and masked models."""

# pylint: disable=consider-using-from-import
import functools
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
import big_vision.evaluators.common as eval_common
import big_vision.input_pipeline as input_pipeline
from big_vision.models.proj.givt import parallel_decode
import big_vision.models.proj.givt.decode as softar_decode
import big_vision.optax as bv_optax
import big_vision.sharding as bv_sharding
import big_vision.trainers.proj.givt.utils as trainer_utils
from big_vision.trainers.proj.uvim import panoptic_task
import big_vision.utils as u
from clu import parameter_overview
import flax
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


NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def main(argv):
  del argv

  jax.distributed.initialize()

  # Make sure TF does not touch GPUs.
  tf.config.set_visible_devices([], "GPU")

  config = flags.FLAGS.config

################################################################################
#                                                                              #
#                                Set up logging                                #
#                                                                              #
################################################################################

  # Set up work directory and print welcome message.
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
  for m in config.get("pp_modules", ["ops_general", "ops_image", "ops_text",
                                     "proj.uvim.pp_ops", "proj.givt.pp_ops"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # Setup up logging and experiment manager.
  xid, wid = -1, -1
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

  # Allow for things like timings as early as possible!
  u.chrono.inform(measure=mw.measure, write_note=write_note)

################################################################################
#                                                                              #
#                                Set up Mesh                                   #
#                                                                              #
################################################################################

  # We rely on jax mesh_utils to organize devices, such that communication
  # speed is the fastest for the last dimension, second fastest for the
  # penultimate dimension, etc.
  config_mesh = config.get("mesh", [("data", jax.device_count())])

  # Sharding rules with default
  sharding_rules = config.get("sharding_rules", [("act_batch", "data")])

  mesh_axes, mesh_size = tuple(zip(*config_mesh))

  # Because jax.utils do not support `-1` shape size.
  mesh_size = np.array(jax.devices()).reshape(mesh_size).shape

  device_mesh = mesh_utils.create_device_mesh(mesh_size)

  # Consistent device order is important to ensure correctness of various train
  # loop components, such as input pipeline, update step, evaluators. The
  # order presribed by the `devices_flat` variable should be used throughout
  # the program.
  devices_flat = device_mesh.flatten()

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
                  steps_per_epoch=ntrain_img / batch_size)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Start input pipeline as early as possible.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_global(train_ds, devices_flat, n_prefetch)

################################################################################
#                                                                              #
#                           Create Model & Optimizer                           #
#                                                                              #
################################################################################

  write_note(f"Creating {config.vae.model_name} model...")
  vae_mod = importlib.import_module(
      f"big_vision.models.{config.vae.model_name}")
  vae = vae_mod.Model(**config.vae.get("model", {}))

  write_note(f"Creating {config.model_name} model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model_config = config.get("model", {})
  model = model_mod.Model(**model_config)

  if config.get("adaptor_name"):
    write_note(f"Creating {config.adaptor_name} model...")
    adaptor_mod = importlib.import_module(
        f"big_vision.models.{config.adaptor_name}")
    adaptor = adaptor_mod.Model(num_channels=model_config.out_dim,
                                **config.adaptor.model)
  else:
    adaptor = None

  def init(rng):
    def _get_dummy_input(input_name, dtype=jnp.int64):
      if input_name in train_ds.element_spec:
        return jnp.zeros(train_ds.element_spec[input_name].shape, dtype=dtype)
      return None

    dummy_img = _get_dummy_input("image", dtype=jnp.float32)
    dummy_labels = _get_dummy_input("labels")
    dummy_cond_img = _get_dummy_input("cond_image", dtype=jnp.float32)
    local_batch_size = dummy_img.shape[0]  # pytype: disable=attribute-error

    code_shape = (
        local_batch_size, model_config.seq_len, model_config.out_dim)
    dummy_code = jnp.zeros(code_shape, jnp.float32)

    input_mask = model.get_input_mask_training(
        jax.random.PRNGKey(0), (local_batch_size, model_config.seq_len)
    )
    params = model.init(rng, dummy_code, dummy_labels, image=dummy_cond_img,
                        input_mask=input_mask)["params"]

    if adaptor is not None:
      _, rng_adaptor = jax.random.split(rng)
      adaptor_variables = adaptor.init(rng_adaptor, dummy_code)
      params_adaptor = flax.core.unfreeze(adaptor_variables["params"])
      params["params_adaptor"] = params_adaptor       # store in same dict

    return params

  rng = jax.random.PRNGKey(u.put_cpu(config.get("seed", 0)))

  write_note("Inferring parameter shapes...")
  rng, rng_init = jax.random.split(rng)
  params_shape = jax.eval_shape(init, rng_init)

  write_note("Inferring optimizer state shapes...")
  tx, sched_fns = bv_optax.make(config, params_shape, sched_kw=dict(
      total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))
  opt_shape = jax.eval_shape(tx.init, params_shape)
  # We jit this, such that the arrays are created on the CPU, not device[0].
  sched_fns_cpu = [u.jit_cpu()(sched_fn) for sched_fn in sched_fns]

  # Training a stage 2 model requires a pretrained stage 1 model. We treat this
  # as a constant and do not shard the parameters.
  assert "model_init" in config.vae
  params_vae = vae_mod.load(None, config.vae.model_init,
                            **config.vae.get("model_load", {}))

  def vae_encode(images, rng=None, reparametrize=True):
    mu, logvar = vae.apply({"params": params_vae}, images, method=vae.encode)
    if reparametrize:
      assert rng is not None and "dropout" in rng
      return vae.apply({"params": params_vae}, mu, logvar,
                       method=vae.reparametrize, rngs=rng)
    return mu

  if jax.process_index() == 0:
    num_params = sum(np.prod(p.shape) for p in jax.tree_leaves(params_shape))
    mw.measure("num_params", num_params)

################################################################################
#                                                                              #
#                               Shard & Transfer                               #
#                                                                              #
################################################################################

  write_note("Creating device mesh...")
  mesh = jax.sharding.Mesh(device_mesh, mesh_axes)
  repl_sharding = jax.sharding.NamedSharding(mesh, P())

  write_note("Inferring shardings...")
  train_state_shape = {"params": params_shape, "opt": opt_shape}

  strategy = config.get("sharding_strategy", [(".*", "replicate")])
  train_state_sharding = bv_sharding.infer_sharding(
      train_state_shape, strategy=strategy, mesh=mesh)

  write_note("Transferring train_state to devices...")
  # RNG is always replicated
  rng_init = u.reshard(rng_init, repl_sharding)

  # Parameters and the optimizer are now global (distributed) jax arrays.
  params = jax.jit(init, out_shardings=train_state_sharding["params"])(rng_init)
  opt = jax.jit(tx.init, out_shardings=train_state_sharding["opt"])(params)

  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)
  del rng  # not used anymore, so delete it.

  # At this point we have everything we need to form a train state. It contains
  # all the parameters that are passed and updated by the main training step.
  train_state = {"params": params, "opt": opt}
  del params, opt  # Delete to avoid memory leak or accidental reuse.

  write_note("Logging parameter overview...")
  parameter_overview.log_parameter_overview(
      train_state["params"], msg="Init params",
      include_stats="global", jax_logging_process=0)

################################################################################
#                                                                              #
#                                 Update Step                                  #
#                                                                              #
################################################################################

  # Define the loss function
  def loss_fn(params, images, labels, cond_images, rng):
    rng, rng_dropout = jax.random.split(rng, 2)
    rng, rng_mask = jax.random.split(rng, 2)
    _, rng_droplabels = jax.random.split(rng, 2)

    rng_dropout = {"dropout": rng_dropout}

    sequence = vae_encode(images, rng_dropout)
    if adaptor is not None:
      # Use the (invertible) adaptor to map to a new latent sequence
      sequence = adaptor.apply({"params": params["params_adaptor"]},
                               sequence, method=adaptor.forward)

    b, s, _ = sequence.shape
    # This is None for the non-mask style. Otherwise, shape (b, s).
    input_mask = model.get_input_mask_training(rng_mask, (b, s))
    drop_labels = model.get_drop_labels(rng_droplabels, batch_size=b)

    _, pdf = model.apply(
        {"params": params}, sequence, labels,
        image=cond_images,
        train=True,
        input_mask=input_mask,
        drop_labels=drop_labels,
        rngs=rng_dropout)

    # Shape: (B, L, out_dim)
    nll = -pdf.log_prob(sequence)
    metrics = {"nll": nll}
    if input_mask is not None:
      metrics["fraction_masked_out"] = input_mask.astype(jnp.float32).mean(
          axis=1
      )
      if nll.ndim == 3:
        input_mask = input_mask[:, :, None]
      # Note that `input_mask` is True where we mask out the input (ie replace
      # with mask token), so we also only gather nlls at the corresponding
      # points.
      nll = jnp.where(input_mask, nll, 0.0)
      # Take mean only of the spots we care about to smooth loss magnitute
      # between examples, like in maskgit (ie this is
      # sum(loss * input_mask) / sum(input_mask) in their code.
      loss = nll.mean(where=input_mask)
    else:
      loss = nll.mean()

    return loss, metrics

  @functools.partial(
      jax.jit,
      donate_argnums=(0,),
      out_shardings=(train_state_sharding, repl_sharding))
  def update_fn(train_state, rng, batch):
    """Update step."""

    images = batch["image"]
    labels, cond_images = batch.get("labels"), batch.get("cond_image")

    step_count = bv_optax.get_count(train_state["opt"], jittable=True)
    rng = jax.random.fold_in(rng, step_count)

    measurements = {}

    # Get device-specific loss rng.
    _, rng_model = jax.random.split(rng, 2)
    params, opt = train_state["params"], train_state["opt"]

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, images, labels, cond_images, rng_model)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)
    train_state = {"params": params, "opt": opt}

    measurements["training_loss"] = loss
    gs = jax.tree_leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree_leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    if adaptor is not None:
      ps_a = jax.tree_leaves(params["params_adaptor"])
      measurements["l2_params_adaptor"] = jnp.sqrt(sum([jnp.vdot(p, p)
                                                        for p in ps_a]))

    measurements.update({f"train/{k}": v.mean() for k, v in metrics.items()})

    return train_state, measurements

################################################################################
#                                                                              #
#                                 Set up Evals                                 #
#                                                                              #
################################################################################

  def validation_fn(train_state, batch, seed=0):
    params = train_state["params"]

    local_rng = trainer_utils.get_local_rng(seed, batch)

    _, aux = loss_fn(
        params, batch["image"], batch.get("labels"),
        batch.get("cond_image"), local_rng)
    return {
        key: jnp.mean(value, axis=tuple(range(1, value.ndim)))
        for key, value in aux.items()
    }

  def predict_fn_teacher_forcing(train_state, batch, seed=0):
    params = train_state["params"]
    image, labels = batch["image"], batch.get("labels")

    local_rng = trainer_utils.get_local_rng(seed, batch)

    rng_dropout = {"dropout": local_rng}
    sequence = vae_encode(image, rng_dropout)

    if adaptor is not None:
      # Use the adaptor to map from VAE latent space to GIVT in/output space.
      sequence = adaptor.apply({"params": params["params_adaptor"]},
                               sequence, method=adaptor.forward)

    b, s, _ = sequence.shape
    # This is None for the non-mask style. Otherwise, shape (b, s) of zeros
    # (nothing masked).
    input_mask = model.get_input_mask_teacher_forced((b, s))

    _, pdf = model.apply(
        {"params": params}, sequence, labels,
        train=True, input_mask=input_mask, rngs=rng_dropout)

    rng_sample, _ = jax.random.split(local_rng, 2)
    sampled_sequence = pdf.sample(seed=rng_sample)

    if adaptor is not None:
      # Use the adaptor inverse to map back to the VAE latent space
      sampled_sequence = adaptor.apply({"params": params["params_adaptor"]},
                                       sampled_sequence, method=adaptor.inverse)
    logits = vae.apply(
        {"params": params_vae}, sampled_sequence, method=vae.decode)

    return {"logits": logits}

  def predict_fn_rep(train_state, image, seed=0):
    assert model.style == "ar"
    assert model.drop_labels_probability == 1.0
    params = train_state["params"]

    local_rng = trainer_utils.get_local_rng(seed, batch)

    rng_dropout = {"dropout": local_rng}
    sequence = vae_encode(image, rng_dropout)
    placeholder_labels = jnp.zeros((sequence.shape[0],), dtype=jnp.int32)

    return model.apply({"params": params}, sequence, labels=placeholder_labels,
                       return_reps=True, method=model.decode)

  def predict_fn_sampling(train_state, batch, seed=0):
    params = train_state["params"]
    labels = batch.get("labels")

    local_rng = trainer_utils.get_local_rng(seed, batch)
    code_logprobs = None

    if model.style == "ar":
      if labels is None:
        # Try to infer batch size if labels are not provided
        if "image" in batch:
          sampling_batch_size = batch["image"].shape[0]
        elif "cond_image" in batch:
          sampling_batch_size = batch["cond_image"].shape[0]
        else:
          sampling_batch_size = config.get("sampling_batch_size", 4)
      else:
        sampling_batch_size = None
      sampled_codes, code_logprobs = softar_decode.generate(
          params={"params": params},
          seed=local_rng,
          model=model,
          seq_len=config.model.seq_len,
          feature_dim=config.model.out_dim,
          labels=labels,
          cond_image=batch.get("cond_image"),
          batch_size=sampling_batch_size,
          config=config.get("ar_generation_config"),
      )
    elif model.style == "masked":
      assert "cond_image" not in batch
      sampled_codes = parallel_decode.decode_masked(  # pytype: disable=wrong-arg-types
          rng=local_rng,
          labels=labels,
          seq_len=config.model.seq_len,
          feature_dim=config.model.out_dim,
          model=model,
          variables={"params": params},
          config=parallel_decode.MaskedGenerationConfig(
              **config.get("masked_generation_config", {})
          ),
      ).current_inputs_q
    else:
      raise NotImplementedError

    if adaptor is not None:
      # Use the adaptor inverse to map back to the VAE latent space.
      sampled_codes = adaptor.apply({"params": params["params_adaptor"]},
                                    sampled_codes, method=adaptor.inverse)

    sampled_images = vae.apply(
        {"params": params_vae}, sampled_codes, method=vae.decode)

    sampling_results = {"logits": sampled_images}
    if code_logprobs is not None:
      sampling_results["logprobs"] = code_logprobs

    return sampling_results

  def predict_fn_sampling_panoptic(
      train_state, batch, seed=0, min_fraction=0.0):
    logits = predict_fn_sampling(train_state, batch, seed)["logits"]
    return panoptic_task.panoptic_predictions_from_logits(
        logits["semantics"], logits["instances"], min_fraction=min_fraction)

  def predict_fn_sampling_depth(train_state, batch, seed=0):
    depth = predict_fn_sampling(train_state, batch, seed)["logits"]["depth"]
    depth = trainer_utils.unbin_depth(
        depth, min_depth=config.min_depth, max_depth=config.max_depth,
        num_bins=config.vae.model.inout_specs["depth"][1])
    return {"depth": depth}

  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config,
        {
            "validation": validation_fn,
            "sample_teacher_forced": predict_fn_teacher_forcing,
            "sample": predict_fn_sampling,
            "sample_panoptic": predict_fn_sampling_panoptic,
            "sample_depth": predict_fn_sampling_depth,
            "representation": predict_fn_rep,
        },
        lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        devices_flat,
    )

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

    # load has the freedom to return params not correctly sharded
    train_state["params"] = u.reshard(
        train_state["params"], train_state_sharding["params"])

    parameter_overview.log_parameter_overview(
        train_state["params"], msg="restored params",
        include_stats="global", jax_logging_process=0)

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
        with mesh, flax.linen.logical_axis_rules(sharding_rules):
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
    # Skip training loop when running an eval-only config
    if config.get("eval_only", False):
      break
    mw.step_start(step)

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with u.chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
        with mesh, flax.linen.logical_axis_rules(sharding_rules):
          train_state, measurements = update_fn(train_state, rng_loop, batch)

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
          with mesh, flax.linen.logical_axis_rules(sharding_rules):
            for key, value in evaluator.run(train_state):
              mw.measure(f"{prefix}{key}", jax.device_get(value))
        u.chrono.resume()
    mw.step_end()

  # Always give a chance to stop the profiler, no matter how things ended.
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
