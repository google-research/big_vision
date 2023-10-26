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

"""Contrastive training loop.

For models Like
- LiT (https://arxiv.org/abs/2111.07991)
- CLIP (https://arxiv.org/abs/2103.00020)
- SigLIP (https://arxiv.org/abs/2303.15343)
"""
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
import big_vision.optax as bv_optax
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


def clip(x, *, a_max=None, a_min=None):
  """Like jnp.clip, but allows all-None to mean don't clip."""
  if a_max is None and a_min is None:
    return x
  return jnp.clip(x, a_max=a_max, a_min=a_min)


def all_gather(z, roll=False, only_others=False):
  """All gather and flatten first two dims."""
  def gather_flat(x):
    x = jax.lax.all_gather(x, "batch")
    if roll or only_others:
      # Each device moves "its" chunk to the beginning. Simplies loss/acc calcs.
      x = jnp.roll(x, -jax.lax.axis_index("batch"), axis=0)
      if only_others:
        x = x[1:]
    return jnp.concatenate(x, 0)  # Fold in "device" and "batch" dims.
  return jax.tree_map(gather_flat, z)


def softmax_loss(zimg, ztxt, temperature):
  """Softmax loss following the CLIP paper. Factorized to reduce memory cost."""

  def unidirectional_loss(z1, z2, t):
    z2 = all_gather(z2, roll=True)
    logits = jnp.dot(z1, z2.T) * t
    # This a softmax across the larger gathered axis, taking advantage of the
    # fact that positives are known to be on the diagonal.
    loss = -(jnp.diag(logits) - jax.scipy.special.logsumexp(logits, axis=-1))
    acc = jnp.argmax(logits, axis=1) == jnp.arange(z1.shape[0])
    return loss.mean(), acc.mean()

  extras = {}
  loss = 0
  for name, row, col in [("i2t", zimg, ztxt), ("t2i", ztxt, zimg)]:
    loss_dir, acc_dir = unidirectional_loss(row, col, temperature)
    loss += 0.5 * loss_dir
    extras[f"{name}_acc"] = acc_dir
    extras[f"{name}_loss"] = loss_dir

  loss = jax.lax.pmean(loss, "batch")
  return loss, extras


def _avg_pos_logit(x_me):
  return jnp.mean(jnp.diag(x_me))


def _avg_neg_logit(x_me, x_ot=None):
  nom = jnp.sum(x_me) - jnp.sum(jnp.diag(x_me))
  den = x_me.size - len(x_me)
  if x_ot is not None and x_ot.size:
    nom += jnp.sum(x_ot)
    den += x_ot.size
  return nom / den


def sigmoid_loss(zimg, ztxt, temperature, bias=0.0):
  """Sigmoid loss from SigLIP: https://arxiv.org/abs/2303.15343."""
  # Sigmoid loss. Since it's unidirectional, image embeddings stick to
  # "me", i.e. the device they are computed on, and text embeddings travel.
  ztxt_me = ztxt  # Text embeddings on my devices: (n, D)
  ztxt_ot = all_gather(ztxt, only_others=True)  # Text emb from others: (N, D)

  logits_me = jnp.dot(zimg, ztxt_me.T)  # (n, D) . (D, n) -> (n, n)
  logits_ot = jnp.dot(zimg, ztxt_ot.T)  # (n, D) . (D, N) -> (n, N)
  logits_me = logits_me * temperature + bias
  logits_ot = logits_ot * temperature + bias

  eye = jnp.eye(zimg.shape[0])
  # Standard sigmoid computes everything twice, once assuming positive
  # labels and once assuming negative ones. But here we know exactly where
  # to find positives (on "me" diagonal) and negatives (everywhere else),
  # so compute each one's loss only once:
  m1_diag1 = -jnp.ones_like(logits_me) + 2 * eye
  loglik_me = jax.nn.log_sigmoid(m1_diag1 * logits_me)
  loglik_ot = jax.nn.log_sigmoid(-logits_ot)

  # Normalize by npos per column, but that's one, so just sum.
  nll_me = -loglik_me.sum(axis=-1)
  nll_ot = -loglik_ot.sum(axis=-1)
  l = nll_me.mean() + nll_ot.mean()  # == concat'ing me/ot along axis -1 above.

  return l, {
      # Only local device metrics for now, as last time I tried, there was
      # some funny unimplemented business with jax.lax.pmin/pmax!
      # So what's reported here is average of per-device min/max/avg.
      "pos_min_logit": jnp.min(jnp.diag(logits_me)),
      "pos_max_logit": jnp.max(jnp.diag(logits_me)),
      "pos_avg_logit": _avg_pos_logit(logits_me),
      "local_neg_min_logit": jnp.min(logits_me + 1e9 * eye),
      "local_neg_max_logit": jnp.max(logits_me - 1e9 * eye),
      "local_neg_avg_logit": _avg_neg_logit(logits_me),
      "neg_min_logit": jnp.minimum(
          jnp.min(logits_me + 1e9 * eye),
          jnp.min(logits_ot) if logits_ot.size else jnp.inf),
      "neg_max_logit": jnp.maximum(
          jnp.max(logits_me - 1e9 * eye),
          jnp.max(logits_ot) if logits_ot.size else -jnp.inf),
      "neg_avg_logit": _avg_neg_logit(logits_me, logits_ot),
  }


def _gather_from_device(x, device_id, axis_name="batch"):
  return jax.lax.psum((jax.lax.axis_index(axis_name) == device_id) * x,
                      axis_name)


def chunked_sigmoid_loss(zimg, ztxt, temperature, bias=0.0):
  """Loss computation from section 3.1 of arxiv.org/abs/2303.15343."""

  # Calculate loss for representations on this device, which includes positives.
  logits_me = jnp.dot(zimg, ztxt.T)  # (n, D) . (D, n) -> (n, n)
  logits_me = logits_me * temperature + bias
  m1_diag1 = -jnp.ones_like(logits_me) + 2 * jnp.eye(zimg.shape[0])
  loglik_me = jax.nn.log_sigmoid(m1_diag1 * logits_me)
  nll_me = -loglik_me.sum(axis=-1).mean()

  def negative_loss(ztxt_other_device):
    logits_ot = jnp.dot(zimg, ztxt_other_device.T)  # (n, D) . (D, n) -> (n, n)
    logits_ot = logits_ot * temperature + bias
    loglik_ot = jax.nn.log_sigmoid(-logits_ot)
    return -jnp.sum(loglik_ot, axis=-1).mean()

  me = jax.lax.axis_index("batch")
  # All other devices are negatives. Hot-potato swap ztxt across devices.
  # Interestingly, ppermute based implementation was memory intensive, so using
  # all-reduce to gather representations.
  nll_others = 0
  for device_id in range(jax.device_count()):
    skip = jnp.not_equal(device_id, me)
    nll_others += skip * negative_loss(_gather_from_device(ztxt, device_id))

  eye = jnp.eye(zimg.shape[0])
  return nll_me + nll_others, {
      "pos_min_logit": jnp.min(jnp.diag(logits_me)),
      "pos_max_logit": jnp.max(jnp.diag(logits_me)),
      "pos_avg_logit": _avg_pos_logit(logits_me),
      "local_neg_min_logit": jnp.min(logits_me + 1e9 * eye),
      "local_neg_max_logit": jnp.max(logits_me - 1e9 * eye),
      "local_neg_avg_logit": _avg_neg_logit(logits_me),}


def main(argv):
  del argv
  tf.config.experimental.set_visible_devices([], "GPU")

  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info(  # pylint: disable=logging-fstring-interpolation
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

  write_note(f"Initializing {config.model_name} model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model = model_mod.Model(**config.get("model", {}))

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend="cpu")
  def init(rng):
    bs = batch_size // jax.device_count()
    image_size = tuple(train_ds.element_spec["image"].shape[1:])
    no_image = jnp.zeros((bs,) + image_size, jnp.float32)
    text_size = tuple(train_ds.element_spec["labels"].shape[1:])
    no_text = jnp.zeros((bs,) + text_size, jnp.int32)
    params = flax.core.unfreeze(model.init(rng, no_image, no_text))["params"]
    return params

  rng, rng_init = jax.random.split(rng)
  with u.chrono.log_timing("z/secs/init"):
    params_cpu = init(rng_init)

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

  @functools.partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
  def update_fn(params, opt, rng, batch):
    """Update step."""
    assert "mixup" not in config, "We still have to figure out mixup."

    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))

    def loss_fn(params, images, labels):
      zimg, ztxt, extras = model.apply(
          {"params": params}, images, labels,
          train=True, rngs={"dropout": rng_model_local})

      match config.get("loss_fn", "softmax"):
        case "softmax":
          l, l_extras = softmax_loss(zimg, ztxt, extras["t"])
        case "sigmoid":
          l, l_extras = sigmoid_loss(zimg, ztxt, extras["t"], bias=extras["b"])
        case "chunked_sigmoid":
          l, l_extras = chunked_sigmoid_loss(zimg, ztxt, extras["t"],
                                             bias=extras["b"])
        case _:
          raise NotImplementedError(f"Unrecognized loss {config.loss_fn=}")

      return l, {
          "t": extras["t"],
          "t/parameter": extras["t/parameter"],
          "train/nimg": jnp.mean(extras["img/norm"]),
          "train/ntxt": jnp.mean(extras["txt/norm"]),
          **{f"train/{k}": v for k, v in l_extras.items()},
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

    return params, opt, rng, l, measurements

  # We require hashable function reference for evaluator.
  # We do not jit/pmap this function, because it is passed to evaluator that
  # does it later. We output as many intermediate tensors as possible for
  # maximal flexibility. Later `jit` will prune out things that are not needed.
  def predict_fn(params, image=None, text=None, **unused_kwargs):
    del unused_kwargs  # `unused_kwargs` is to be compatible with few-shot
    zimg, ztxt, out = model.apply({"params": params}, image, text)
    return zimg, ztxt, out

  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config, {"predict": predict_fn},
        lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
    )

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
        "chrono": u.chrono.save(),
    }
    checkpoint_tree = jax.tree_structure(checkpoint)
    loaded = u.load_checkpoint_np(resume_ckpt_path, checkpoint_tree)
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    checkpoint = jax.tree_map(u.recover_dtype, loaded)
    params_cpu, opt_cpu = checkpoint["params"], checkpoint["opt"]
    u.chrono.load(checkpoint["chrono"])
  elif config.get("model_init"):
    write_note(f"Initialize model from {config.model_init}...")
    params_cpu = model_mod.load(
        params_cpu, config.model_init, config.get("model"),
        **config.get("model_load", {}))
    if jax.process_index() == 0:
      parameter_overview.log_parameter_overview(
          params_cpu, msg="restored params")

  write_note("Kicking off misc stuff...")
  first_step = bv_optax.get_count(opt_cpu)
  u.chrono.inform(first_step=first_step)
  prof = None  # Keeps track of start/stop of profiler state.

  write_note(f"Replicating...\n{u.chrono.note}")
  params_repl = flax.jax_utils.replicate(params_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  ckpt_writer = None

  write_note(f"First step compilations...\n{u.chrono.note}")

  # Note that training can be pre-empted during the final evaluation (i.e.
  # just after the final checkpoint has been written to disc), in which case we
  # want to run the evals.
  if first_step in (total_steps, 0):
    mw.step_start(first_step)
    for (name, evaluator, _, prefix) in evaluators():
      if config.evals[name].get("skip_first") and first_step != total_steps:
        continue
      write_note(f"{name} evaluation...\n{u.chrono.note}")
      with u.chrono.log_timing(f"z/secs/eval/{name}"):
        for key, value in evaluator.run(params_repl):
          mw.measure(f"{prefix}{key}", value)

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with u.chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
        params_repl, opt_repl, rngs_loop, loss_value, measurements = update_fn(
            params_repl, opt_repl, rngs_loop, batch)

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
      u.chrono.pause(wait_for=(params_repl, opt_repl))
      u.checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see (internal link)). Also, takes device 0's params only.
      params_cpu = jax.tree_map(lambda x: np.array(x[0]), params_repl)
      opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if u.itstime(step, get_steps("keep_ckpt", None), total_steps):
        copy_step = step

      ckpt = {"params": params_cpu, "opt": opt_cpu, "chrono": u.chrono.save()}
      ckpt_writer = pool.apply_async(
          u.save_checkpoint, (ckpt, save_ckpt_path, copy_step))
      u.chrono.resume()

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
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
