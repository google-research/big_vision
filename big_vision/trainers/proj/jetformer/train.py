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

"""Training loop for JetFormer."""
# pylint: disable=consider-using-from-import
# pylint: disable=logging-fstring-interpolation

import functools
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
import big_vision.evaluators.common as eval_common
import big_vision.input_pipeline as input_pipeline
import big_vision.optax as bv_optax
import big_vision.sharding as bv_sharding
import big_vision.trainers.proj.jetformer.predict_fns as predict_fns
import big_vision.utils as u
from clu import parameter_overview
import distrax
import flax
import flax.linen as nn
import jax
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serial
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf

from tensorflow.io import gfile


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

  # This is needed on multihost systems, but crashes on non-TPU single-host.
  if os.environ.get("BV_JAX_INIT"):
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

  write_note("Creating device mesh...")
  mesh = u.create_device_mesh(
      config_mesh,
      allow_split_physical_axes=config.get("mesh_allow_split_physical_axes",
                                           False))
  repl_sharding = jax.sharding.NamedSharding(mesh, P())

  # Consistent device order is important to ensure correctness of various train
  # loop components, such as input pipeline, update step, evaluators. The
  # order presribed by the `devices_flat` variable should be used throughout
  # the program.
  devices_flat = mesh.devices.flatten()

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

  assert config.patch_pca.model_name == "proj.jetformer.patch_pca", (
      "This trainer only supports proj.jetformer.patch_pca as an embedder."
  )
  write_note(f"Creating {config.patch_pca.model_name} model...")
  pca_mod = importlib.import_module(
      f"big_vision.models.{config.patch_pca.model_name}")
  patch_pca = pca_mod.Model(**config.patch_pca.get("model", {}))

  # Apply PCA - does not require any learnable parameters.
  def patch_pca_encode(images, rng=None, reparametrize=True):
    mu, logvar = patch_pca.apply(
        {"params": {}}, images, method=patch_pca.encode, rngs=rng)
    if reparametrize:
      assert rng is not None and "dropout" in rng
      return patch_pca.apply({"params": {}}, mu, logvar,
                             method=patch_pca.reparametrize, rngs=rng)
    return mu

  write_note(f"Creating {config.model_name} model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model_config = config.get("model", {})
  model = model_mod.Model(**model_config)

  if config.get("adaptor_name"):
    write_note(f"Creating {config.adaptor_name} model...")
    adaptor_mod = importlib.import_module(
        f"big_vision.models.{config.adaptor_name}")
    adaptor = adaptor_mod.Model(**config.adaptor.model)
  else:
    adaptor = None

  def adaptor_apply(params, sequence, inverse=False):
    # Apply NVP and ensure compatible input/output format.
    sequence = predict_fns.unflatten_latents(sequence)
    assert hasattr(adaptor, "forward") and hasattr(adaptor, "inverse")
    sequence, sum_log_det = adaptor.apply(
        {"params": params}, sequence,
        method=adaptor.inverse if inverse else adaptor.forward)
    sequence = predict_fns.flatten_latents(sequence)
    return sequence, sum_log_det

  def _maybe_remove_latent_noise_dims(image_tokens):
    if (noise_dim := config.get("latent_noise_dim", 0)) > 0:
      image_tokens = image_tokens[..., :-noise_dim]
      assert image_tokens.shape[-1] == model.out_dim
    return image_tokens

  def init(rng, batch=None):
    # TODO: Update init function with new arguments
    def _get_dummy_input(input_name, dtype=jnp.int64):
      if batch is not None:
        return batch.get(input_name)
      elif input_name in train_ds.element_spec:
        return jnp.zeros(train_ds.element_spec[input_name].shape, dtype=dtype)
      return None

    images = _get_dummy_input("image", dtype=jnp.float32)
    text = _get_dummy_input("text")
    assert images is not None and text is not None

    image_tokens = patch_pca_encode(images, rng={"dropout": rng})

    if adaptor is not None:
      rng, rng_adaptor = jax.random.split(rng)
      image_tokens = predict_fns.unflatten_latents(image_tokens)
      (image_tokens, _), adaptor_variables = adaptor.init_with_output(
          rng_adaptor, image_tokens, method=adaptor.forward)
      params_adaptor = flax.core.unfreeze(adaptor_variables["params"])
      image_tokens = predict_fns.flatten_latents(image_tokens)
    else:
      params_adaptor = {}

    image_tokens = _maybe_remove_latent_noise_dims(image_tokens)

    text_first = jnp.full(images.shape[0], 0)
    params = model.init(rng, text, image_tokens,
                        text_input_mask=_get_dummy_input("text_mask"),
                        text_first_mask=text_first)["params"]
    params["params_adaptor"] = params_adaptor
    return params

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  # See (internal link) for a fun read on what it takes.
  rng = jax.random.PRNGKey(u.put_cpu(config.get("seed", 0)))

  write_note("Inferring parameter shapes...")
  rng, rng_init = jax.random.split(rng)
  params_shape = jax.eval_shape(init, rng_init)

  write_note("Inferring optimizer state shapes...")
  tx, sched_fns = bv_optax.make(config, nn.unbox(params_shape), sched_kw=dict(
      total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))
  opt_shape = jax.eval_shape(tx.init, params_shape)
  # We jit this, such that the arrays are created on the CPU, not device[0].
  sched_fns_cpu = [u.jit_cpu()(sched_fn) for sched_fn in sched_fns]

  if jax.process_index() == 0:
    num_params = sum(np.prod(p.shape) for p in jax.tree.leaves(params_shape))
    mw.measure("num_params", num_params)

################################################################################
#                                                                              #
#                               Shard & Transfer                               #
#                                                                              #
################################################################################

  write_note("Inferring shardings...")
  train_state_shape = {"params": params_shape, "opt": opt_shape}

  if config.get("ema_decay", 0.0) > 0.0:
    write_note(f"Tracking parameter EMA with decay {config.ema_decay}.")
    train_state_shape["params_ema"] = params_shape

  strategy = config.get("sharding_strategy", [(".*", "replicate")])
  with nn.logical_axis_rules(sharding_rules):
    train_state_sharding = bv_sharding.infer_sharding(
        train_state_shape, strategy=strategy, mesh=mesh)

  write_note("Transferring train_state to devices...")
  # RNG is always replicated
  rng_init = u.reshard(rng_init, repl_sharding)

  # Parameters and the optimizer are now global (distributed) jax arrays.
  first_batch = next(train_iter)
  params = jax.jit(init, out_shardings=train_state_sharding["params"])(
      rng_init, first_batch)
  opt = jax.jit(tx.init, out_shardings=train_state_sharding["opt"])(params)

  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)
  del rng  # not used anymore, so delete it.

  train_state = {"params": params, "opt": opt}
  if config.get("ema_decay", 0.0) > 0.0:
    # Copy model parameters for EMA
    train_state["params_ema"] = jax.tree.map(jnp.array, train_state["params"])

  # At this point we have everything we need to form a train state. It contains
  # all the parameters that are passed and updated by the main training step.
  # From here on, we have no need for Flax AxisMetadata (such as partitioning).
  train_state = nn.unbox(train_state)
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
  def loss_fn(params, batch, rng, noise_scale=None, train=True):
    text, images = batch["text"], batch["image"]
    text_mask, text_loss = batch["text_mask"], batch["text_loss"]

    rng, rng_dropout, rng_order, rng_droplabels, rng_noise = (
        jax.random.split(rng, 5))

    rng_dropout = {"dropout": rng_dropout}

    batch_size = images.shape[0]
    # 0 -> image first, 1 -> text first
    text_first_mask = jax.random.bernoulli(
        rng_order, config.get("text_prefix_prob", 0.5), (batch_size,))

    if noise_scale is not None:
      # Maybe skip noise on image prefix.
      if not config.get("rgb_noise_on_image_prefix", True):
        noise_scale = jnp.where(text_first_mask, noise_scale, 0.0)
        noise_scale = noise_scale[:, None, None, None]  # [bs, h, w, 3]
      # Convert images to [0, 255] add noise with std scale round to int and
      # convert back to [-1, 1]. This way it is as if the input was added
      # to the uint8 images in the preprocessing before the value_range(-1, 1).
      images = jnp.round((images+1)*127.5)
      images = images + noise_scale * jax.random.normal(rng_noise, images.shape)
      images = jnp.round(images)
      images = images/127.5 - 1

    image_tokens = patch_pca_encode(images, rng_dropout)
    if adaptor is not None:
      # Use the (invertible) adaptor to map to a new latent sequence
      image_tokens, sum_log_det = adaptor_apply(
          params["params_adaptor"], image_tokens)
    else:
      sum_log_det = jnp.zeros((images.shape[0],),)

    if (noise_dim := config.get("latent_noise_dim", 0)) > 0:
      # Mapping the last noise_dim dimensions to a standard normal prior.
      assert model.out_dim + noise_dim == image_tokens.shape[-1]
      image_tokens, noise = jnp.split(image_tokens, [model.out_dim], axis=-1)
      noise_pdf = distrax.Normal(0.0, 1.0)
      noise_nll = -noise_pdf.log_prob(noise).sum(axis=(1, 2))
    else:
      noise_nll = 0.0

    if train and (input_noise_std := config.get("input_noise_std", 0.0)) > 0.0:
      # Add noise on the input during teacher forcing to make autoregressive
      # sampling more robust: Sample a noise std uniformly at random per example
      # and add Gaussian noise with that std to the input.
      _, rng_std, rng_input_noise = jax.random.split(rng, 3)
      sampled_input_noise_std = jax.random.uniform(
          rng_std, (batch_size, 1, 1), minval=0.0, maxval=input_noise_std)
      # Only apply noise for image generation (i.e. when text is first).
      sampled_input_noise_std = jnp.where(
          text_first_mask[:, None, None], sampled_input_noise_std, 0.0)
      image_tokens = image_tokens + (
          sampled_input_noise_std
          * jax.random.normal(rng_input_noise, image_tokens.shape))

    # TODO: Do cfg for text and don't apply prefix loss when dropped.
    # For now only drop when text is first.
    if train:
      drop_prefix = model.get_drop_labels(rng_droplabels, batch_size=batch_size)
    else:
      drop_prefix = None
    if drop_prefix is None:
      drop_prefix = jnp.full((batch_size,), False)
    drop_prefix = drop_prefix & text_first_mask

    # Stop gradients to NVP when it is used as an encoder to get an image prefix
    if config.get("stop_grad_nvp_prefix", False):
      image_tokens = jnp.where(
          text_first_mask[:, None, None],
          image_tokens,
          jax.lax.stop_gradient(image_tokens)
      )

    *_, pmf, pdf, _ = model.apply(
        {"params": params},
        text,
        image_tokens,
        train=train,
        text_first_mask=text_first_mask,
        text_input_mask=text_mask,
        drop_prefix=drop_prefix,
        rngs=rng_dropout)

    def _log_prob(value):
      # Re-implementing distrax.Categorical.log_prob() without logic to ignore
      # NaNs, which is not required and seems to produce a compilation error
      # with recent jax versions.
      value_one_hot = jax.nn.one_hot(
          value, pmf.num_categories, dtype=pmf.logits.dtype)
      mask_outside_domain = jnp.logical_or(
          value < 0, value > pmf.num_categories - 1)
      return jnp.where(
          mask_outside_domain, -jnp.inf,
          jnp.sum(pmf.logits * value_one_hot, axis=-1))
    nll_txt = -_log_prob(text)  # [BS, TXT_LEN]
    nll_txt = jnp.mean(nll_txt, axis=1, where=text_loss)

    # Report image related loss in log2/subpixels (bits per subpixels).
    # When using PCA this value is off, as it does not accounts for logdet
    # of PCA and ignores the perplexity of dropped PCA components.
    num_subpixels = np.prod(images.shape[1:])  # H*W*C
    nll_image_tokens = -pdf.log_prob(image_tokens)  # [BS, IMG_LEN]
    nll_image_tokens = (
        jnp.sum(nll_image_tokens, axis=1) + noise_nll) / num_subpixels
    nll_image_tokens /= jnp.log(2)
    # Convert logdet sum to be per subpixel and account for conversion
    # [0, 255]->[-1, 1] (i.e. by divide by 127.5).
    logdet = sum_log_det / num_subpixels - jnp.log(127.5)
    logdet /= jnp.log(2)
    nll_image = nll_image_tokens - logdet

    def mean(x, where=None):
      if valid_example_mask := batch.get("_mask", None) is not None:
        if where is not None:
          where = where & valid_example_mask
        else:
          where = valid_example_mask
      return jnp.mean(x, where=where)

    metrics = {
        "nll_text_prefix": mean(
            nll_txt, where=text_first_mask & ~drop_prefix),
        "nll_text_suffix": mean(nll_txt, where=~text_first_mask),
        # Currently, we never drop the image prefix, but we already consider
        # this case here.
        "nll_image_prefix": mean(
            nll_image, where=~text_first_mask & ~drop_prefix),
        "nll_image_suffix": mean(nll_image, where=text_first_mask),
    }

    text_w = config.get("text_loss_weight", 1.0)
    if config.get("loss_on_prefix", True):
      valid_txt_nll = (text_first_mask & ~drop_prefix) | ~text_first_mask
      valid_img_nll = (~text_first_mask & ~drop_prefix) | text_first_mask
      metrics.update({
          "nll_text": mean(nll_txt, where=valid_txt_nll),
          "nll_image": mean(nll_image, where=valid_img_nll),
          "logdet": mean(logdet),
      })
      loss = (mean(nll_txt, where=valid_txt_nll) * text_w
              + mean(nll_image, where=valid_img_nll))
    else:
      text_suffix = ~text_first_mask
      image_suffix = text_first_mask
      metrics.update({
          "nll_text": mean(nll_txt, where=text_suffix),
          "nll_image": mean(nll_image, where=image_suffix),
          "nll_image_tokens": mean(nll_image_tokens, where=image_suffix),
          "logdet": mean(logdet, where=image_suffix),
      })
      example_loss = jnp.where(text_suffix, nll_txt*text_w, nll_image)
      loss = mean(example_loss)

    metrics["loss"] = loss
    return loss, metrics

  @functools.partial(
      jax.jit,
      donate_argnums=(0,),
      out_shardings=(train_state_sharding, repl_sharding))
  def update_fn(train_state, rng, batch):
    """Update step."""
    step_count = bv_optax.get_count(train_state["opt"], jittable=True)
    rng = jax.random.fold_in(rng, step_count)

    measurements = {}
    progress = step_count / total_steps

    if config.get("noise_scale", 0.0) > 0.0:
      noise_min = config.get("noise_min", 0.0)
      noise_scale = ((config.noise_scale - noise_min)
                     * (1+jnp.cos(jnp.pi*progress)) * 0.5) + noise_min
      measurements["noise_scale"] = noise_scale
    else:
      noise_scale = None

    # Get device-specific loss rng.
    _, rng_model = jax.random.split(rng, 2)
    params, opt = train_state["params"], train_state["opt"]

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch, rng_model, noise_scale=noise_scale)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)
    new_train_state = {"params": params, "opt": opt}
    # Update EMA parameters.
    if (ema_decay := config.get("ema_decay", 0.0)) > 0.0:
      new_params_ema = jax.tree.map(
          lambda pe, p: ema_decay * pe + (1 - ema_decay) * p,
          train_state["params_ema"], params)
      new_train_state["params_ema"] = new_params_ema

    measurements["training_loss"] = loss
    gs = jax.tree.leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree.leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree.leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    if adaptor is not None:
      ps_a = jax.tree.leaves(params["params_adaptor"])
      measurements["l2_params_adaptor"] = jnp.sqrt(sum([jnp.vdot(p, p)
                                                        for p in ps_a]))

    measurements.update({f"train/{k}": v.mean() for k, v in metrics.items()})

    return new_train_state, measurements

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
    jax.tree.map(lambda x: x.delete(), train_state)
    del train_state
    shardings = {
        **train_state_sharding,
        "chrono": jax.tree.map(lambda _: repl_sharding,
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


################################################################################
#                                                                              #
#                                 Setup Evals                                  #
#                                                                              #
################################################################################

  def validation_fn(train_state, batch, *, use_ema=False):
    params = train_state["params_ema"] if use_ema else train_state["params"]
    rng = jax.random.PRNGKey(
        jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))

    _, aux = loss_fn(params, batch, rng, train=False)
    # The metrics produced by loss_fn may already be averaged over the batch,
    # since some of them only apply for certain batches. Here we broadcast the
    # metrics across the batch dimension, which might introduce some errors when
    # the batch size is not divisible by the number of devices.
    aux = jax.tree.map(
        lambda x: jnp.broadcast_to(x, batch["text"].shape[:1]), aux)
    return aux

  def sample_images_fn(train_state, batch, *, decode_len=256, use_ema=False):
    params = train_state["params_ema"] if use_ema else train_state["params"]
    cfg_weight = config.sample_images.get("cfg_inference_weight", 0.0)
    temperature = config.sample_images.get("temperature", 1.0)
    temperature_probs = config.sample_images.get("temperature_probs", 1.0)

    if batch["text"].ndim < 2:
      batch["text"] = batch["text"][:, None]

    out = predict_fns.sample_image_latents(
        params, batch, model=model, decode_len=decode_len,
        cfg_weight=cfg_weight, temperature=temperature,
        temperature_probs=temperature_probs)

    image_tokens = out["out_tokens"]
    if (noise_dim := config.get("latent_noise_dim", 0)) > 0:
      rng = jax.random.PRNGKey(
          jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))
      noise = jax.random.normal(rng, image_tokens.shape[:-1] + (noise_dim,))
      image_tokens = jnp.concatenate([image_tokens, noise], axis=-1)

    images = predict_fns.decode_images(
        params["params_adaptor"], image_tokens,
        adaptor=adaptor, patch_pca=patch_pca)
    out["logits"] = images
    return out

  def sample_text_fn(params, batch, *, temperature, decode_len):
    """Jittable sampling of text."""
    image_latents = predict_fns.encode_images(
        params["params_adaptor"], batch["image"],
        adaptor=adaptor, patch_pca=patch_pca,
        rngs={"dropout": jax.random.key(0)},
        reparametrize=False)

    image_latents = _maybe_remove_latent_noise_dims(image_latents)

    out = predict_fns.sample_text(
        params, {"image_latents": image_latents, **batch}, model=model,
        temperature=temperature, decode_len=decode_len)
    return out["out_tokens"]

  def sample_text(train_state, batch, *,
                  use_ema=False, decode_len, temperature=1e-5, devices,
                  eos_token=None):
    """Predict fn that does the jitting of sample_text_fn for evaluators."""
    del eos_token  # Unused. We always sample decode_len tokens.
    params = train_state["params_ema"] if use_ema else train_state["params"]
    mesh = jax.sharding.Mesh(devices, ("devices",))
    data_sharding = jax.sharding.NamedSharding(mesh, P("devices"))
    new_batch = {
        "image": batch["image"],
        "text": batch.get("text", None),
        "text_mask": batch.get("text_mask", None),
    }
    tokens = jax.jit(sample_text_fn, out_shardings=data_sharding,
                     static_argnames=("decode_len", "temperature"))(
                         params, new_batch,
                         decode_len=decode_len, temperature=temperature)
    return tokens

  def score_captions_fn(train_state, batch, *, use_ema=False):
    # TODO: Enable caching of the prefix to speed up evaluation.
    params = train_state["params_ema"] if use_ema else train_state["params"]
    images = batch["image"]
    all_labels = batch["_label_tokens"]
    all_labels_mask = batch["_label_masks"]
    all_loss_masks = batch["_loss_masks"]
    batch_size = images.shape[0]

    rng = jax.random.PRNGKey(
        jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))
    rng_dropout = {"dropout": rng}
    image_tokens = patch_pca_encode(images, rng_dropout)

    if adaptor is not None:
      image_tokens, _ = adaptor_apply(params["params_adaptor"], image_tokens)

    image_tokens = _maybe_remove_latent_noise_dims(image_tokens)

    def _score_label(label_and_mask):
      label, mask, loss_mask = label_and_mask
      label_rep = jnp.tile(label, (batch_size, 1))
      masks_rep = jnp.tile(mask, (batch_size, 1))
      loss_masks_rep = jnp.tile(loss_mask, (batch_size, 1))
      _, _, pmf, *_ = model.apply(
          {"params": params},
          text_tokens=label_rep,
          image_tokens=image_tokens,
          text_first_mask=jnp.full((batch_size,), False),  # images always first
          text_input_mask=masks_rep,
      )
      return jnp.sum(pmf.log_prob(label_rep), axis=-1, where=loss_masks_rep)

    ll_labels = jax.lax.map(
        _score_label, (all_labels, all_labels_mask, all_loss_masks))
    return ll_labels.T

  def image_rep_fn(train_state, batch, *, use_ema=False):
    params = train_state["params_ema"] if use_ema else train_state["params"]
    images = batch["image"]

    rng = jax.random.PRNGKey(
        jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))
    rng_dropout = {"dropout": rng}
    image_tokens = patch_pca_encode(images, rng_dropout)

    out = {"patch_emb": image_tokens}
    if adaptor is not None:
      image_tokens, _ = adaptor_apply(params["params_adaptor"], image_tokens)
      out["nvp"] = image_tokens

    image_tokens = _maybe_remove_latent_noise_dims(image_tokens)

    # TODO: Allow the code to be called without text tokens and think
    # better what representations to use here... For now its the representation
    # obtained by feeding the 257 tokens [BOI, image_tokens] to the model. The
    # text tokens are not used. It is only [BOS] but since its the last token in
    # the input it is not feed to the model (dropped by shift).
    *_, decoder_out = model.apply(
        {"params": params},
        text_tokens=jnp.full((batch_size, 0,), 0),  # [BS, 0]: empty text.
        image_tokens=image_tokens,
        text_first_mask=jnp.full((batch_size,), False),  # images always first
    )
    out.update(decoder_out)

    # Average pool intermediate representations.
    out = jax.tree.map(lambda x: x.mean(axis=-2), out)

    return out["pre_logits"], out

  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config,
        {
            "validation": validation_fn,
            "sample_images": sample_images_fn,
            "sample_text": sample_text,
            "score_captions": score_captions_fn,
            "image_representation": image_rep_fn,
        },
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
        with mesh, nn.logical_axis_rules(sharding_rules):
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
        with mesh, nn.logical_axis_rules(sharding_rules):
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
      chrono_shardings = jax.tree.map(lambda _: repl_sharding, chrono_ckpt)
      ckpt = ckpt | {"chrono": u.reshard(chrono_ckpt, chrono_shardings)}

      u.save_checkpoint_ts(ckpt_mngr, ckpt, save_ckpt_path, step, keep)
      u.chrono.resume()

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
        u.chrono.pause(wait_for=train_state)
        u.chrono.tick(step)  # Record things like epoch number, core hours etc.
        write_note(f"{name} evaluation...\n{u.chrono.note}")
        with u.chrono.log_timing(f"z/secs/eval/{name}"):
          with mesh, nn.logical_axis_rules(sharding_rules):
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
