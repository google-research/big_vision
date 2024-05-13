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

"""Utils very specific to this project, not generic."""

import collections
import contextlib
import dataclasses
import functools
import io
import json
import multiprocessing
import multiprocessing.pool
import os
import re
import sys
import time
from typing import Mapping

from absl import flags
from absl import logging
from big_vision.pp import registry as pp_registry
import einops
import flax
import flax.jax_utils as flax_utils
import jax
from jax.experimental.array_serialization import serialization as array_serial
import jax.numpy as jnp
import ml_collections as mlc
import numpy as np

import tensorflow.io.gfile as gfile  # pylint: disable=consider-using-from-import


Registry = pp_registry.Registry


# pylint: disable=logging-fstring-interpolation


def pad_shard_unpad(wrapped, static_argnums=(0,), static_argnames=()):
  """Wraps a function with code that pads, shards, then un-shards, un-pads.

  Args:
    wrapped: the function to be wrapped. Signature is `params, *args, *kwargs`.
    static_argnums: indices of arguments to `wrapped` that should _not_ be
      padded and sharded, but instead be forwarded as-is. The default is (0,)
      because by far the most common use-case is to pass `params` first.
    static_argnames: names of kwargs to `wrapped` that should _not_ be padded
      and sharded, but instead be forwarded as-is.

  Returns:
    A new function that pads and shards its arguments before passing them to
    the wrapped function, and un-shards and un-pads the returned pytree.

    This is useful for calling a pmap'ed function with inputs that aren't
    divisible by the number of devices. A typical use is:
      @pad_shard_unpad
      @jax.pmap
      def forward(params, x): ...

  Notes:
    The padding is done in host-memory before being passed to the function, and
    the values returned by the function are transferred back to host memory.

    The returned function is augmented with a new keyword-only argument
    `min_device_batch` that, if specified, forces padding inputs to at least
    this size per device. This can be useful to avoid recompiles for the last
    batch and reduce memory fragmentation.
  """

  def pad_shard_unpad_wrapper(*args, min_device_batch=None, **kw):
    d = jax.local_device_count()  # d = devices, b = batch

    # Find the batch-sizes of all non-static arguments.
    def get_bs(x):
      batch_sizes = jax.tree.map(lambda y: y.shape[0], x)
      return jax.tree.flatten(batch_sizes)[0]

    bs_a = [get_bs(a) for i, a in enumerate(args) if i not in static_argnums]
    bs_kw = [get_bs(v) for k, v in kw.items() if k not in static_argnames]
    bs = set([n for b in (bs_a + bs_kw) for n in b])
    assert len(bs) == 1, f"Inconsistent batch-sizes: {bs}"
    b = bs.pop()

    def pad(x):
      _, *shape = x.shape
      db, rest = divmod(b, d)
      if rest:
        x = np.concatenate([x, np.zeros((d - rest, *shape), x.dtype)], axis=0)
        db += 1
      if min_device_batch and db < min_device_batch:
        x = np.concatenate(
            [x, np.zeros((d * (min_device_batch - db), *shape), x.dtype)])
        db = min_device_batch
      return x.reshape(d, db, *shape)

    def maybe_pad(x, actually_pad=True):
      if not actually_pad: return x  # For call-site convenience below.
      return jax.tree.map(pad, x)

    args = [maybe_pad(a, i not in static_argnums) for i, a in enumerate(args)]
    kw = {k: maybe_pad(v, k not in static_argnames) for k, v in kw.items()}
    out = wrapped(*args, **kw)

    def unpad(x):
      # Transfer back before cutting, to reduce on-device shape diversity.
      return einops.rearrange(jax.device_get(x), "d b ... -> (d b) ...")[:b]
    return jax.tree.map(unpad, out)

  return pad_shard_unpad_wrapper


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  x = jax.lax.select(x, jnp.full(x.shape, on_value),
                     jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  # Load the data; use local paths directly if possible:
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    # For other (remote) paths go via gfile+BytesIO as np.load requires seeks.
    with gfile.GFile(fname, "rb") as f:
      data = f.read()
    loaded = np.load(io.BytesIO(data), allow_pickle=False)

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)


def load_checkpoint_np(npz, tree=None):
  """Loads a jax pytree from a npz file.

  Args:
    npz: Either path to the checkpoint file (.npz), or a dict-like.
    tree: deprecated, use None.
      Bwd-compat for old format that only stored values: the pytree structure.

  Returns:
    A pytree that is the checkpoint.
  """
  if isinstance(npz, str):  # If not already loaded, then load.
    npz = npload(npz)
  keys, values = zip(*list(npz.items()))
  if tree:
    checkpoint = tree.unflatten(values)
  else:
    checkpoint = recover_tree(keys, values)
  return checkpoint


def load_params(ckpt, **kw):
  """Loads the parameters of a big_vision checkpoint, both old or new format.

  Args:
    ckpt: Path to the checkpoint (.npz, .ts) or dict-like.
    **kw: forwarded to the underlying load function (_np or _ts).

  Returns:
    A pytree that is the checkpoint, potentially sharded.

  Notes:
    The `ckpt` string can contain an colon-separated "submodel" indicator, like
    `img` in the example `/path/to/file.npz:img`.
    This is used to load sub-parts of a model, for example the image load the
    image encoder out of a two_tower (SigLIP) checkpoint, or distillation.
    This way, ANY model that uses this function can load itself from a
    checkpoint that contains multiple sub-models.
  """
  key = None  # Whether we want to extract only a sub-key of the model.

  if isinstance(ckpt, str):  # Most common case of passing a checkpoint path.
    # Potentially read out the sub-part to load from after the colon
    # '/path/to/file:img/head' => '/path/to/file', 'img/head'
    # 'gs://path/to/file' => 'gs://path/to/file', None
    if match := re.match(r"^(.*?/.*?)(?::([\w/]+))?$", ckpt):
      ckpt, key = match.groups()
    else:
      raise ValueError(f"Weird ckpt path: {ckpt} ; Maybe prepend ./ ?")

    # Use the checkpoint filename to detect when we're loading old-style .npz
    # checkpoints, as opposed to new-style tensorstore checkpoint folders.
    if ".npz" in ckpt:  # Not a perfect heuristic, but good enough.
      checkpoint = load_checkpoint_np(ckpt, **kw)
      checkpoint = jax.tree.map(recover_dtype, checkpoint)
      if "params" in checkpoint:
        # Checkpoint with optax state (after (internal link)).
        params = checkpoint["params"]
      elif "opt" in checkpoint:
        # Checkpoint with Flax optimizer.
        params = checkpoint["opt"]["target"]
      else:
        # When open-sourcing, we often shared only the params directly.
        params = checkpoint
    else:
      # Here we're now loading new-style tensorstore checkpoints.
      # We can be a more efficient and load params and `key` only right away.
      regex = f"params/{key}($|/.*)" if key else "params/.*"
      checkpoint = load_checkpoint_ts(ckpt, regex=regex)
      params = checkpoint["params"]

  if key is not None:
    params = tree_get(params, key)

  return params


def prefetch_scalar(it, nprefetch=1, devices=None):
  n_loc_dev = len(devices) if devices else jax.local_device_count()
  repl_iter = (np.ones(n_loc_dev) * i for i in it)
  return flax_utils.prefetch_to_device(repl_iter, nprefetch, devices)


def sigmoid_xent(*, logits, labels, reduction=True):
  # NOTE: This implementation is stable, see these two:
  # (internal link)
  # https://github.com/google/jax/issues/2140
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  nll = -jnp.sum(labels * log_p + (1. - labels) * log_not_p, axis=-1)
  return jnp.mean(nll) if reduction else nll


def bidirectional_contrastive_loss(zimg, ztxt, t, mask=None, reduction=False):
  """Bidirectional contrastive loss (e.g. for contrastive trainer/evaluator)."""
  # BF.FB = BB
  logits = jnp.dot(zimg, ztxt.T) * t

  if mask is not None:
    # Set to negative infinity where mask = 0. Masked examples will disappear
    # under softmax, and be ignored by ncorrect (NINF will never win argmax).
    exclude = jnp.logical_not(mask)  # Now 1 if we don't want to keep.
    exclude = jnp.logical_or(exclude[:, None], exclude[None, :])
    logits = jnp.where(exclude, -jnp.inf, logits)

  # Note: assumed t is in a good range e.g. already passed through exp/softplus.
  l1 = -jnp.diag(jax.nn.log_softmax(logits, axis=1))  # NLL img->txt
  l2 = -jnp.diag(jax.nn.log_softmax(logits, axis=0))  # NLL txt->img
  l = 0.5 * (l1 + l2)

  if mask is not None:
    l = jnp.where(mask, l, 0)

  redux = jnp.mean if reduction else lambda x: x
  if reduction and mask is not None:
    redux = lambda x: jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)

  # Also return extra measurements.
  return redux(l), {
      "ncorrect": redux(jnp.argmax(logits, axis=1) == jnp.arange(len(logits))),
  }


def softmax_xent(*, logits, labels, reduction=True, kl=False, axis=-1):
  log_p = jax.nn.log_softmax(logits, axis=axis)
  nll = -jnp.sum(labels * log_p, axis=axis)
  if kl:
    nll += jnp.sum(labels * jnp.log(jnp.clip(labels, 1e-8)), axis=axis)
  return jnp.mean(nll) if reduction else nll


def weighted_softmax_xent(*,
                          logits,
                          labels,
                          reduction=True,
                          weights=None,
                          label_smoothing=0.0,
                          normalize=True):
  """Compute weighted cross entropy.

  Args:
   logits: [batch, length, num_classes] float array.
   labels: categorical targets [batch, length] int array.
   reduction: reduce across batch dim.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   normalize: normalize each "sentence" loss by the number of tokens in it.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != labels.ndim + 1:
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(labels.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  soft_targets = onehot(
      labels, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)

  normalizing_factor = labels.shape[1]
  if weights is not None:
    loss = loss * weights
    normalizing_factor = jnp.clip(weights.sum(axis=1), 2e-38)

  loss = loss.sum(axis=1)
  if normalize:
    loss = loss / normalizing_factor

  return loss.mean() if reduction else loss


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
  """Accumulate gradient over multiple steps to save on memory."""
  # See (internal link) for details and experiments.
  if accum_steps and accum_steps > 1:
    assert images.shape[0] % accum_steps == 0, (
        f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}")
    step_size = images.shape[0] // accum_steps
    l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])
    def acc_grad_and_loss(i, l_and_g):
      imgs = jax.lax.dynamic_slice(images, (i*step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i*step_size, 0),
                                   (step_size, labels.shape[1]))
      li, gi = loss_and_grad_fn(params, imgs, lbls)
      l, g = l_and_g
      return (l + li, jax.tree.map(lambda x, y: x + y, g, gi))
    l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
    return jax.tree.map(lambda x: x / accum_steps, (l, g))
  else:
    return loss_and_grad_fn(params, images, labels)


def itstime(step, every_n_steps, total_steps, host=None, last=True, first=True,
            drop_close_to_last=0.25):
  """Returns True if it's time to execute an action.

  Args:
    step: the current step representing "now".
    every_n_steps: the action should run every this many steps.
    total_steps: the step number of the last step of training.
    host: host number. If provided, only run if we are this process.
    last: whether to run on the last step or not.
    first: whether to run on the first step or not.
    drop_close_to_last: if a step would run, but is this close (in terms of
      fraction of every_n_step) to the last one, skip.

  Returns:
    True if the action should be executed, False if not.
  """

  # This logic avoids running `itstime` "a few" steps before the last step.
  # Canonical example: don't save checkpoint 2 steps before the last, and then
  # at the last again; it's pointless and checkpoint timing will time out.
  close_to_last = False
  if drop_close_to_last and every_n_steps:
    close_to_last = abs(step - total_steps) < drop_close_to_last * every_n_steps

  is_host = host is None or jax.process_index() == host
  is_step = every_n_steps and (step % every_n_steps == 0) and not close_to_last
  is_last = every_n_steps and step == total_steps
  is_first = every_n_steps and step == 1
  return is_host and (is_step or (last and is_last) or (first and is_first))


def checkpointing_timeout(writer, timeout):
  # Make sure checkpoint writing is not a bottleneck
  if writer is not None:
    try:
      # Note: `writer` is a multiprocessing.AsyncResult, and
      # timeout is in seconds.
      writer.get(timeout=timeout)
    except multiprocessing.TimeoutError as e:
      raise TimeoutError(
          "Checkpoint writing seems to be a bottleneck. Make sure you do "
          "not do something wrong, like writing checkpoints to a distant "
          "cell. In a case you are OK with checkpoint writing being a "
          "bottleneck, you can configure `ckpt_timeout` parameter") from e


def hms(s):
  """Format time in hours/minutes/seconds."""
  if s < 60:
    return f"{s:.0f}s"
  m, s = divmod(s, 60)
  if m < 60:
    return f"{m:.0f}m{s:.0f}s"
  h, m = divmod(m, 60)
  if h < 25:
    return f"{h:.0f}h{m:.0f}m"  # Seconds intentionally omitted.
  d, h = divmod(h, 24)
  return f"{d:.0f}d{h:.0f}h{m:.0f}m"  # Seconds intentionally omitted.


class Chrono:
  """Measures time and reports progress, hyper-specific to our train loops.

  Some concepts:
  1. This differentiates between three "types" of time:
    - training time: the time spent on actual training (fprop/bprop/update)
    - program time: overall time the program runs, including all overheads
    - pause time: the chronometer can be paused (eg during evals).
  2. This handles a "warmup": the first step is skipped for training time
      purposes, as it includes significant compilation overheads, which distort
      estimates.
  3. `accum`ulates (i.e. integrates) timings, and save/load them across
      restarts.
  """

  def __init__(self):
    self._timing_history = collections.defaultdict(list)
    self._measure = None
    self._write_note = None

    self.program_start_time = time.monotonic()
    self.train_start_time = None
    self.train_start_step = None  # When we started timing (after warmup)

    self.prev_time = None
    self.prev_step = None

    self.pause_start = None
    self.paused_time = 0

    self.total_steps = None
    self.global_bs = None
    self.steps_per_epoch = None

    self.warmup = 2  # How many calls to `tick` to skip.
    self.load()  # Inits accum integrators.
    self.note = "Chrono n/a"

  def inform(self, *, first_step=None, total_steps=None, global_bs=None,
             steps_per_epoch=None, measure=None, write_note=None):
    """Provide some extra info that's only known later in the program."""
    # The pattern of `self.x = x or self.x` allows one to call `inform` various
    # times with various subset of information (args), as they become available.
    # Except for `first_step` which can be 0 so is a bit more verbose.
    self.prev_step = first_step if first_step is not None else self.prev_step
    self.total_steps = total_steps or self.total_steps
    self.steps_per_epoch = steps_per_epoch or self.steps_per_epoch
    self.global_bs = global_bs or self.global_bs
    self._measure = measure or self._measure
    self._write_note = write_note or self._write_note
    if self.total_steps and self.prev_step is not None:
      self.note = (f"Steps:{self.prev_step}/{self.total_steps} "
                   f"[{self.prev_step/self.total_steps:.1%}]")

  def tick(self, step, measure=None, write_note=None):
    """A chronometer tick."""
    if step == self.prev_step: return  # Can happen from evals for example.

    measure = measure or self._measure
    write_note = write_note or self._write_note

    now = time.monotonic()
    measure("uptime", now - self.program_start_time)
    self.flush_timings()

    # We do always count examples, regardless of the timing-related warmup that
    # happens a few lines below.
    ds = step - self.prev_step  # Steps between ticks
    self.prev_step = step
    self.accum_examples_seen += ds * self.global_bs
    measure("examples_seen", self.accum_examples_seen)
    measure("progress", step / self.total_steps)
    if self.steps_per_epoch:
      measure("epoch", step / self.steps_per_epoch)

    # We take the start as the second time `tick` is called, so we avoid
    # measuring the overhead of compilation and don't include it in time
    # estimates.
    if self.warmup > 1:
      self.warmup -= 1
      write_note(self.note)  # This can help debugging.
      return
    if self.warmup == 1:
      self.train_start_time = self.prev_time = now
      self.train_start_step = step
      self.accum_program_time += now - self.program_start_time
      self.paused_time = 0  # Drop pauses that happened before timing starts.
      self.warmup = 0
      write_note(self.note)  # This can help debugging.
      return

    # Measurement with micro-timings of current training steps speed.
    # Time between ticks (ignoring pause)
    dt = now - self.prev_time - self.paused_time
    ncores = jax.device_count()  # Global device count
    measure("img/sec/core", self.global_bs * ds / dt / ncores)

    # Accumulate (integrate) times, good for plots.
    self.accum_train_time += dt
    self.accum_pause_time += self.paused_time
    self.accum_program_time += dt + self.paused_time

    # Convert to, and log as, core hours.
    core_hours = self.accum_train_time * ncores / 60 / 60
    devtype = jax.devices()[0].device_kind
    measure(f"core_hours_{devtype}", core_hours)
    measure("core_hours", core_hours)  # For convenience as x-axis in sweeps.

    # Progress note with "global" full-program average timings
    # (eg in program-time minus warmup)
    dt = now - self.train_start_time  # Time elapsed since end of warmup.
    steps_timed = step - self.train_start_step
    steps_todo = self.total_steps - step
    self.note = f"Steps:{step}/{self.total_steps} [{step/self.total_steps:.1%}]"
    self.note += f"\nWalltime:{hms(self.accum_program_time)}"
    self.note += f" ({hms(self.accum_pause_time)} eval)"
    self.note += f"\nETA:{hms(dt / steps_timed*steps_todo)}"
    self.note += f"\nTotal train time:{hms(dt / steps_timed*self.total_steps)}"
    write_note(self.note)

    self.prev_time = now
    self.paused_time = 0

  def pause(self, wait_for=()):
    assert self.pause_start is None, "Don't pause twice."
    jax.block_until_ready(wait_for)
    self.pause_start = time.monotonic()

  def resume(self):
    self.paused_time += time.monotonic() - self.pause_start
    self.pause_start = None

  def save(self):
    return dict(
        accum_program_time=self.accum_program_time,
        accum_train_time=self.accum_train_time,
        accum_pause_time=self.accum_pause_time,
        accum_examples_seen=self.accum_examples_seen,
    )

  def load(self, ckpt={}):  # pylint: disable=dangerous-default-value
    self.accum_program_time = float(ckpt.get("accum_program_time", 0.0))
    self.accum_train_time = float(ckpt.get("accum_train_time", 0.0))
    self.accum_pause_time = float(ckpt.get("accum_pause_time", 0.0))
    self.accum_examples_seen = int(ckpt.get("accum_examples_seen", 0))

  @contextlib.contextmanager
  def log_timing(self, name, *, noop=False):
    """Use this when you time sth once per step and want instant flushing."""
    t0 = time.monotonic()
    yield
    dt = time.monotonic() - t0
    if not noop:
      if self._measure:  # So that timed things still work in colab.
        self._measure(name, dt)
      logging.info("TIMING[%s]: %s", name, dt)
      logging.flush()

  @contextlib.contextmanager
  def log_timing_avg(self, name, *, noop=False):
    """Use this when you time sth multiple times per step (eg in a loop)."""
    t0 = time.monotonic()
    yield
    dt = time.monotonic() - t0
    if not noop:
      self._timing_history[name].append(dt)
      logging.info("TIMING[%s]: avg %s current %s",
                   name, np.mean(self._timing_history[name]), dt)
      logging.flush()

  def flush_timings(self):
    assert self._measure is not None
    for name, times in self._timing_history.items():
      self._measure(name, np.mean(times))
    self._timing_history.clear()


# Singleton to use from everywhere. https://stackoverflow.com/a/6760726/2366315
chrono = Chrono()


def _traverse_with_names(tree, with_inner_nodes=False):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree.map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key], with_inner_nodes):
        yield (key + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  elif isinstance(tree, (list, tuple)):
    for idx in range(len(tree)):
      for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
        yield (str(idx) + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  else:
    yield "", tree


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names is provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree.flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_unflatten(names_and_vals):
  """Reverses `tree_flatten_with_names(tree)[0]`."""
  return recover_tree(*zip(*names_and_vals))


def tree_map_with_names(f, tree, *rest):
  """Like jax.tree.map but with a filter on the leaf path name.

  Args:
    f: A function with first parameter `name` (path-like "a/b/c") and remaining
      parameters values of `tree` and `*rest` corresponding to the given `name`
      Should return a new value for parameter `name`.
    tree: The tree of parameters `f` should be applied to.
    *rest: more trees of the exact same structure.

  Returns:
    A tree identical in structure to `tree` and `*rest` but with the leaves the
    result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
  """
  names_and_vals, tree_def = tree_flatten_with_names(tree)
  names, vals = zip(*names_and_vals)
  rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
  vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
  return tree_def.unflatten(vals)


def tree_map_with_regex(f, tree, regex_rules, not_f=lambda x: x, name=None):
  """Apply jax-style tree_map based on regex rules.

  Args:
    f: a function that is being applied to every variable.
    tree: jax tree of arrays.
    regex_rules: a list of tuples `(pattern, args)`, where `pattern` is a regex
      which used for variable matching and `args` are positional arguments
      passed to `f`. If some variable is not matched, we apply `not_f` transform
      which is id by default. If multiple patterns match, then only the first
      rule is applied.
    not_f: optional function which is applied to variables that do not match any
      pattern.
    name: a name of transform for logging purposes.

  Returns:
    a tree, transformed by `f` according to the given rules.
  """
  def _f(vname, v):
    for pattern, arg in regex_rules:
      if re.fullmatch(pattern, vname):
        if name and jax.process_index() == 0:
          logging.info("Applying %s to %s with %s due to `%s`",
                       name, vname, arg, pattern)
        return f(v, arg)
    return not_f(v)
  return tree_map_with_names(_f, tree)


def tree_get(tree, name):
  """Get an entry of pytree by flattened key name, eg a/b/c, with nice error.

  Args:
    tree: the pytree to be queried.
    name: the path to extract from the tree, see below for examples.

  Returns:
    A few examples:
      tree = {'a': 1, 'b': {'c': 2, 'd': 3}}
      tree_get(tree, 'a') == 1
      tree_get(tree, 'b/c') == 2
      tree_get(tree, 'b') == {'c': 2, 'd': 3}
  """
  flattened = dict(_traverse_with_names(tree, with_inner_nodes=True))
  try:
    return flattened[name]
  except KeyError as e:
    class Msg(str):  # Reason: https://stackoverflow.com/a/70114007/2366315
      def __repr__(self):
        return str(self)
    msg = "\n".join([name, "Available keys:", *flattened, ""])
    # Turn into configdict to use its "did you mean?" error message!
    msg = mlc.ConfigDict(flattened)._generate_did_you_mean_message(name, msg)  # pylint: disable=protected-access
    raise KeyError(Msg(msg)) from e


def tree_replace(tree, replacements):
  """Renames/removes (nested) keys.

  Example usage:

    tree = {'a': {'b': 2, 'c': 3}, 'c': 4}
    replacements = {
        'a/b': 'a/b/x',  # replaces 'a/b' with 'a/b/x'
        '.*c': 'C',      # replaces 'c' with 'C' ('a/c' is removed)
        'C': 'D',        # replaces 'C' (which was 'c') with 'D'
        '.*/c': None,    # removes 'a/c'
    }
    tree2 = rename_remove(tree, replacements)
    assert tree2 == {'D': 4, 'a': {'b': {'x': 2}}}

  Args:
    tree: A nested dictionary.
    replacements: Rules specifying `regex` as keys and `replacement` as values
      to be used with `m = re.match(regex, key)` and `m.expand(replacement)`
      for every `key` independently.

      Note that:
      1. If any rule matches with `replacement=None`, then the key is removed.
      2. The rules are applied in order. It's possible to have multiple
         transformations on a single key.

  Returns:
    Updated `tree` according to rules defined in `replacements`.
  """
  replacements = {
      re.compile(kk): vv for kk, vv in replacements.items()
  }

  def rename(k):
    for kk, vv in replacements.items():
      m = kk.match(k)
      if m:
        k = k[:m.start()] + m.expand(vv) + k[m.end():]
    return k

  def should_remove(k):
    return any(vv is None and kk.match(k) for kk, vv in replacements.items())

  names_and_vals, _ = tree_flatten_with_names(tree)
  names_and_vals = [
      (rename(k), v) for k, v in names_and_vals if not should_remove(k)
  ]
  return tree_unflatten(names_and_vals)


def tree_compare(tree1, tree2):
  """Returns `(tree1_only, tree2_only, dtype_shape_mismatch)`."""
  tree1 = flax.traverse_util.flatten_dict(tree1, sep="/")
  tree2 = flax.traverse_util.flatten_dict(tree2, sep="/")
  return set(tree1) - set(tree2), set(tree2) - set(tree1), {
      k: [(v.dtype, v.shape), (tree2[k].dtype, tree2[k].shape)]
      for k, v in tree1.items()
      if k in tree2 and (v.dtype != tree2[k].dtype or v.shape != tree2[k].shape)
  }


def tree_filter(tree, mask):
  """Returns nested dict structure with only a subset of children."""
  # TODO: The code below only works for nested-dict and only when they
  # have same structure. Consider relax this.
  if not isinstance(tree, dict):
    assert isinstance(mask, bool), f"Mask leaves must be boolean! {mask}"
    return tree
  assert sorted(tree.keys()) == sorted(mask.keys()), (
      f"Keys in tree and mask are not equal! {tree.keys()} != {mask.keys()}")
  return {k: tree_filter(v, mask[k]) for k, v in tree.items()
          if mask[k] is not False}


def recover_dtype(a):
  """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
  if hasattr(a, "dtype") and a.dtype.type is np.void:
    assert a.itemsize == 2, "Unknown dtype!"
    return a.view(jax.numpy.bfloat16)
  else:
    return a


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.

  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def tssave(mngr, pytree, path, on_commit=lambda *_, **__: None):
  """Save pytree using jax tensorstore-based checkpoint manager.

  NOTE: When overwriting an existing checkpoint with a different pytree, the
  result is, counterintuitively, the union of both, not only the new one.

  Args:
    mngr: An instance of GlobalAsyncCheckpointManager.
    pytree: What to store; any pytree of arrays.
    path: Where to save the pytree. Creates subfolders as needed.
    on_commit: A callback when writing is done, see `mngr.serialize`.
  """
  names, vals = zip(*tree_flatten_with_names(pytree)[0])

  for name in names:
    if "~" in name:
      raise ValueError(f"Symbol '~' is not allowed in names. Found in {name}.")

  gfile.makedirs(path)
  with jax.transfer_guard("allow"):
    names = [name.replace("/", "~") for name in names]
    mngr.serialize_with_paths(
        list(vals), [os.path.join(path, name) for name in names],
        on_commit_callback=functools.partial(on_commit, array_names=names))


def save_checkpoint_ts(mngr, checkpoint, path, step, keep=True):
  """Preemption-safe saving of checkpoints using tssave."""
  # The tensorstore checkpoint format is a folder with (potentially) many files.
  # On some file-systems, operations on these (copy, rename, delete) are slow,
  # so we implement a flow that's both robust to pre-emptions/crashes during
  # checkpointing and makes minimal use of these slow operations.

  # The logic goes as follows. It's infaillible :)
  #       (...if file move is atomic, which it is.)
  # We always write the current checkpoint to a new folder, which contains the
  # step number in its name. If we don't need to keep it indefinitely, we append
  # "-tmp" to its name.
  # After writing the next checkpoint, we remove the previous one if it had
  # "-tmp" in its name.
  # We also have a -LAST file that contains a pointer to the latest complete
  # checkpoint. File operations are cheap to make atomic, that's why.

  def _on_commit_callback(array_names):  # Runs after writing ckpt is done.
    with gfile.GFile(f"{path}-CUR", "w") as f:
      f.write(curr)

    last = ""
    if gfile.exists(f"{path}-LAST"):
      with gfile.GFile(f"{path}-LAST", "r") as f:
        last = f.read().strip()

    gfile.rename(f"{path}-CUR", f"{path}-LAST", overwrite=True)

    if last.endswith("-tmp"):
      # If pre-emption happens here, some old checkpoints may not be deleted.
      multiprocessing.pool.ThreadPool().map(
          gfile.rmtree,
          [f"{path}-{last}/{name}" for name in array_names])
      gfile.rmtree(f"{path}-{last}")

  # NOTE: The jax checkpoint manager automatically waits for the previous save
  # to be finished before writing again, so we don't need to do it here.

  # Always write to path with step number in it.
  curr = f"{step:09d}{'-tmp' if not keep else ''}"
  tssave(mngr, checkpoint, f"{path}-{curr}", _on_commit_callback)


def load_checkpoint_ts(path, **tsload_kw):
  """Loads a big_vision checkpoint saved by `save_checkpoint_ts`."""
  to_load = path

  try:
    # When passing a general path (not a specific step), get the last available.
    with gfile.GFile(f"{path}-LAST", "r") as f:
      to_load = f"{path}-{f.read().strip()}"
  except Exception:  # Differs based on backend, so blanket catch. pylint:disable=broad-exception-caught
    pass

  return tsload(to_load, **tsload_kw)


def tsload(path, *, tree=None, shardings=None, regex=None):
  """Loads tensorstore-based array-tree from disk.

  If `tree` argument is provided, then array names to load and target structure
  is derived from the tree. If `tree` is None, then array names to load are
  derived from array filenames on the disk, and, optionally, `regex` is applied
  to filter these names. The`tree` argument is then automatically derived from
  array names with `recover_tree` util.

  Arrays are loaded to CPU/TPU/GPU memory as specified by the `shardings`
  argument, which is a pytree of CPU/TPU/GPU shardings (can be mixed within a
  single pytree). `shardings` should a prefix tree of the `tree` argument. We
  automatically broadcast `shardings` to a full `tree`. For example, a user can
  specify `shardings=jax.sharding.SingleDeviceSharing(jax.devices('cpu')[0])`,
  which  will be broadcasted to a full tree.

  Args:
    path: a directory where the checkpoint arrays are stored.
    tree: a target pytree, which defines array names to load and the target tree
      structure. If tree is None, then `tree` is inferred from the names of
      arrays stored on the disk.
    shardings: a prefix pytree (with respect to `tree`) of the target shardings.
    regex: regex to filter array names from the disk, if `tree` is not provided.

  Returns:
    A pytree of loaded arrays that has the same structure as `shardings` arg.
  """
  if (tree is not None) and (regex is not None):
    raise ValueError("If tree is specified, regex filtering is not allowed.")

  if tree is None:
    # Some file-systems (gs://) list folders with a trailing /, get rid of it.
    path_names = set([p.rstrip("/").replace("~", "/")
                      for p in gfile.listdir(path)])
    regex = re.compile(regex) if regex is not None else re.compile(".*")
    path_names = [p for p in path_names if regex.match(p)]
    tree = recover_tree(path_names, [0] * len(path_names))

  names_and_vals, tree_def = tree_flatten_with_names(tree)
  names_to_load, _ = zip(*names_and_vals)

  if shardings is None:
    shardings = jax.sharding.SingleDeviceSharding(
        jax.local_devices(backend="cpu")[0]
    )
  shardings = list(jax.tree.leaves(tree_broadcast(shardings, tree)))

  names_to_load = [os.path.join(path, name.replace("/", "~"))
                   for name in names_to_load]
  specs = [array_serial.get_tensorstore_spec(n) for n in names_to_load]
  arrays = array_serial.run_deserialization(shardings, specs)
  return tree_def.unflatten(arrays)


def steps(prefix, config, data_size=None, batch_size=None, total_steps=None,
          default=ValueError):
  """Gets duration named `prefix` out of `config` and converts it to steps.

  Using this function to access a configuration value that denotes some kind
  of duration (eg training time, warmup, checkpoint frequency, ...) allows the
  duration to be specified in terms of steps, epochs, examples, or percent of
  training time, and converts any of these into steps, such that the training
  code only deals with steps.
  If the result is not an integer step number, it is rounded to the nearest one.

  Args:
    prefix: The name of the duration to query. The actual config fields can
      then be one of `prefix_steps`, `prefix_examples`, or `prefix_epochs`.
    config: The dictionary (config) from which to read the duration.
    data_size: The total number of training examples in one epoch.
    batch_size: The number of examples processed per step.
    total_steps: The total number of training steps to run.
    default: The default value to return when no duration of the name `prefix`
      is found in the `config`. Set to `ValueError` (the default) to raise an
      error instead of returning a default value.

  Returns:
    The number of steps from the config, or the default value.

  Raises:
    ValueError if there is no such duration in the config and no default is set.
  """
  # Be helpful and make sure only match one of the following suffixes.
  suffixes = {"steps", "examples", "epochs", "percent"}
  matches = {f"{prefix}_{s}" for s in suffixes if f"{prefix}_{s}" in config
             and config[f"{prefix}_{s}"] is not None}
  # Note that steps=0 is also a valid value (e.g. to only run evaluators).
  assert len(matches) <= 1, f"Only one of '{matches}' should be defined."

  if f"{prefix}_steps" in config:
    return config[f"{prefix}_steps"]

  def to_integer(x):
    # Round to nearest but always executed at least one step unless explictily
    # asked for 0. E.g. total_epochs=0 vs total_epochs=0.0001
    return max(1, round(x)) if x else 0

  if batch_size and f"{prefix}_examples" in config:
    return to_integer(config[f"{prefix}_examples"] / batch_size)

  if batch_size and data_size and f"{prefix}_epochs" in config:
    steps_per_epoch = data_size / batch_size
    return to_integer(config[f"{prefix}_epochs"] * steps_per_epoch)

  if total_steps and f"{prefix}_percent" in config:
    pct = config[f"{prefix}_percent"]
    assert 0.0 <= pct <= 1.0, (  # Be helpful, since it's not obvious.
        f"Percents should lie in [0.0, 1.0], but {prefix}_percent is {pct}")
    return to_integer(pct * total_steps)

  if default is ValueError:
    raise ValueError(
        f"Cannot convert {prefix} to steps, due to missing batch_size "
        f"({batch_size}), data_size ({data_size}), total_steps ({total_steps})"
        ", or corresponding entry in config:\n" + "\n".join(config.keys()))

  return default


def create_learning_rate_schedule(
    total_steps, batch_size=None, data_size=None,
    base=1.0, decay_type="stair",
    scale_with_batchsize=False, **kw):
  """Creates learning rate schedule, see (internal link).

  Args:
    total_steps: The total number of steps to run.
    batch_size: The global batch-size optionally used for scaling.
    data_size: Number of examples in the training data (for epoch conversion).
    base: The starting learning-rate (without warmup).
    decay_type: 'linear' or 'cosine', 'rsqrt', 'stair'.
    scale_with_batchsize: Whether or not to scale lr automatically.
    **kw: extra arguments specific to individual decay_types. Also contains
      declaration of `{warmup,cooldown}_{steps,epochs,examples}` that applies
      on top of any/all decay_type.

  Returns:
    A function learning_rate(step): float -> {"learning_rate": float}.
  """

  warmup_steps = steps(
      "warmup", kw, data_size, batch_size, total_steps, default=0)
  cooldown_steps = steps(
      "cooldown", kw, data_size, batch_size, total_steps, default=0)

  # Early catch hard to backtrack errors due to warmup_steps >= total_steps,
  # but let it run for 0 and 1 steps used to eval and debug runs.
  assert (total_steps <= 1) or (warmup_steps < total_steps), (
      "warmup_steps is >= total_steps")

  def step_fn(step):
    """Step to learning rate function."""
    lr = base

    # This implements the linear scaling rule following
    # Goyal et al. at arxiv.org/abs/1706.02677.
    # The reference batch size in literature is 256, so we scale the lr to
    # adjust to the literature lr when bach_size changes.
    if scale_with_batchsize:
      lr = lr * batch_size / 256.0

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = jnp.clip(progress, 0.0, 1.0)
    if decay_type in ("linear", "polynomial"):
      power = kw.get("power", 1)
      zero = kw.get("end", kw.get("linear_end", 0))
      lr = zero + (lr - zero) * (1.0 - progress) ** power
    elif decay_type == "cosine":
      lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
    elif decay_type == "rsqrt":
      # See (internal link) for details, especially how to set timescale
      # and shift in order to continue smoothly when changing batch-size.
      if "timescale_examples" in kw:
        t = kw["timescale_examples"] / batch_size
      else:
        t = kw.get("timescale", 10_000)  # bwd-compat default.
      shift = kw.get("shift", 0)
      lr = jnp.where(
          warmup_steps <= step,
          lr / jnp.sqrt(1 + (step + shift - warmup_steps) / t),  # In decay
          lr / jnp.sqrt(1 + shift / t))  # In warmup.
    elif decay_type == "stair":
      i = jnp.searchsorted(jnp.array(kw.get("steps", [])), step + 1)
      lr = lr * jnp.take(jnp.array([1.0] + list(kw.get("mults", []))), i)
    else:
      raise ValueError(f"Unknown lr type {decay_type}")

    if warmup_steps:
      lr = lr * jnp.minimum(1., step / warmup_steps)
    if cooldown_steps:
      lr = lr * jnp.minimum(1., (total_steps - step) / cooldown_steps)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn


def get_mixup(rng, p):
  """Perform mixup https://arxiv.org/abs/1710.09412."""
  rng, rng_mixup = jax.random.split(rng)
  a = jax.random.beta(rng_mixup, p, p)
  a = jnp.maximum(a, 1.0 - a)  # see (internal link) for the context.
  def _mixup(*things, **more_things):
    mix = lambda thing: a * thing + (1 - a) * jnp.roll(thing, shift=1, axis=0)
    return rng, *jax.tree.map(mix, (things, more_things))
  return _mixup


# For backwards compatability with legacy code.
def mixup(rng, *things, p, **more_things):
  return get_mixup(rng, p)(*things, **more_things)


def sync():
  """Syncs hosts and empties async computation queue."""
  x = reshard(np.ones(jax.device_count()),
              jax.sharding.PositionalSharding(jax.devices()))
  jax.jit(jnp.sum)(x).block_until_ready()


def check_and_compile_patterns(patterns):
  """Validates and compiles a list of param-patterns.

  The validation consists of checking for common mistakes, currently only that
  the pattern does not start with a slash, because unlike FLAX, our parameter
  names don't start with a slash.

  Args:
    patterns: a single (string) pattern (regex), or a list of patterns.

  Returns:
    A list of compiled and verified regexes.
  """
  if isinstance(patterns, str):
    patterns = [patterns]

  assert isinstance(patterns, (list, tuple)), patterns

  def check_and_compile(pattern):
    assert not pattern.startswith("/"), (
        f"Big vision parameter names never start with '/': '{pattern}")
    return re.compile(pattern)

  return list(map(check_and_compile, patterns))


def make_mask_trees(tree, patterns, *, log=None):
  """Returns a boolean mask tree for every pattern (only first match)."""
  compiled_patterns = check_and_compile_patterns(patterns)

  def matchfirst(name, _):
    matches = []
    for pattern in compiled_patterns:
      matches.append(not any(matches) and bool(pattern.fullmatch(name)))
    if log is not None and True in matches and jax.process_index() == 0:
      logging.info("%s: %s - matched by %s", log, name,
                   patterns[matches.index(True)])
    return np.array(matches)

  multimask = tree_map_with_names(matchfirst, tree)
  return [
      jax.tree.map(lambda matches, i=idx: matches[i], multimask)
      for idx in range(len(patterns))
  ]


@contextlib.contextmanager
def profile(name, ttl=3 * 365 * 24 * 3600, noop=False):
  if not noop:
    sess = startstop_prof_at_steps(None, name=name, ttl=ttl)
  yield
  if not noop:
    startstop_prof_at_steps(sess, name=name, ttl=ttl)


def startstop_prof(sess, step=None, first_step=0,
                   log_steps=1, surround=20, **kw):
  """Runs the profiler for `surround` steps around the next `log_steps`."""
  first_log = first_step + log_steps - (first_step % log_steps)
  # don't start before first!
  start = max(first_log - surround//2, first_step + 1)
  return startstop_prof_at_steps(sess, step, start, start + surround, **kw)


def startstop_prof_at_steps(
    sess, step=None, first_step=None, last_step=None,
    name="steps", ttl=3 * 365 * 24 * 3600):
  del sess, step, first_step, last_step, name, ttl
  pass  # TODO: implement using `jax.profiler` API. Needs workdir.


# This is a very minimal variant for open-sourcing. Our internal code makes use
# of multiple internal logging tools instead.
class BigVisionMetricWriter:
  """A class for logging metrics."""

  def __init__(self, xid=-1, wid=-1, workdir=None, config=None):
    self.step_start(0)
    if jax.process_index() != 0: return  # Only one host shall write stuff.

    self.pool = multiprocessing.pool.ThreadPool(1)  # 1 is important here.
    self.fname = None
    if workdir:
      if xid != -1 and wid != -1:
        self.fname = os.path.join(workdir,
                                  f"big_vision_{xid}_{wid}_metrics.txt")
      else:
        self.fname = os.path.join(workdir, "big_vision_metrics.txt")
      if config:
        with gfile.GFile(os.path.join(workdir, "config.json"), "w") as f:
          f.write(config.to_json())

  def step_start(self, step):
    self.step = step
    self.step_metrics = {}

  def measure(self, name, value):
    """Logs the metric value."""
    if jax.process_index() != 0: return  # Only one host shall write stuff.

    # Convenience for accepting scalar np/DeviceArrays, as well as N-d single
    # scalars, like [[[123]]] or similar, avoiding silly mistakes.
    value = np.array(value).squeeze()

    # If the value is a scalar, we keep it in mind to append a line to the logs.
    # If it has any structure, we instead just log its shape.
    value = float(value) if value.ndim == 0 else value.shape

    logging.info(f"\u001b[35m[{self.step}]\u001b[0m {name} = {value}")
    logging.flush()
    self.step_metrics[name] = value

    return value  # Just for convenience

  def step_end(self):
    """Ends a training step, write its full row."""
    if not self.step_metrics: return

    def write(metrics):
      with gfile.GFile(self.fname, "a") as f:
        f.write(json.dumps({"step": self.step, **metrics}) + "\n")

    if self.fname:
      self.pool.apply(lambda: None)  # Potentially wait for past writes.
      self.pool.apply_async(write, (self.step_metrics,))

  def close(self):
    self.step_end()
    if jax.process_index() == 0:
      self.pool.close()
      self.pool.join()


def maybe_cleanup_workdir(workdir, cleanup, info):
  """Potentially removes workdirs at end of run for cleanup."""
  if not workdir:
    return

  if not cleanup:
    info("Logs/checkpoints are in %s", workdir)
  elif jax.process_index() == 0:
    gfile.rmtree(workdir)
    try:  # Only need this on the last work-unit, if already empty.
      gfile.remove(os.path.join(workdir, ".."))
    except tf.errors.OpError:
      pass


def tree_broadcast(prefix, target):
  """Broadcasts a prefix tree to a full tree.

  Input-output examples:
  1. prefix: {"x": 10, "y": 20}
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: {"x": {"a": 10, "b": 10}, "y": 20}

  2. prefix: 100
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: {"x": {"a": 100, "b": 100}, "y": 100}

  3. prefix: {"x": 10}
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: ValueError

  Args:
    prefix: prefix pytree.
    target: boradcast target for a prefix tree.

  Returns:
    prefix tree broadcasted to a target tree.
  """
  def _broadcast(leaf, subtree):
    return jax.tree.map(lambda _: leaf, subtree)
  return jax.tree.map(_broadcast, prefix, target)


def reshard(tree, shardings):
  """Take an arbitrarily* sharded pytree and shard it according to `shardings`.

  This is a no-op for tree elements which are already sharded as requested.

  *Arrays that are fully addressable (for example, CPU arrays) are assumed to be
  identical (i.e. replicated) across hosts.

  *It does not work if an element of `tree` is not fully-addressable, unless its
  sharding is already consistent with the target sharding.
  If this is needed, please ping lbeyer@ or akolesnikov@.

  Args:
    tree: a pytree of arrays.
    shardings: a (prefix) pytree of jax array shardings.
  Returns:
    A pytree of global jax arrays that follows provided shardings.
  """
  def _make_global_arr(x, shard, shape):
    # Avoid unnecessary copies and transfers:
    if hasattr(x, "sharding") and x.sharding.is_equivalent_to(shard, len(shape)):  # pylint: disable=line-too-long
      return x
    if not getattr(x, "is_fully_addressable", True):
      raise RuntimeError("Trying to reshard a non-fully-addressable array. "
                         "Please see the doc-comment for detailed explanation.")
    x = jax.device_get(x)  # Might be on local devices.
    xs = [jax.device_put(x[s], device=d)
          for d, s in shard.addressable_devices_indices_map(shape).items()]
    return jax.make_array_from_single_device_arrays(shape, shard, xs)

  shapes = jax.tree.map(np.shape, tree)
  shardings = tree_broadcast(shardings, tree)
  return jax.tree.map(_make_global_arr, tree, shardings, shapes)


def put_cpu(x):
  """Places array/pytree on a CPU device."""
  return jax.device_put(x, jax.local_devices(backend="cpu")[0])


def make_fsarray_from_local_slice(local_slice, global_devices):
  """Create a fully-sharded global device array from local host arrays.

  Args:
    local_slice: Something convertible to a numpy array (eg also TF tensors)
      that is this host's slice of the global array.
    global_devices: The list of global devices. Needed for consistent ordering.

  Returns:
    The global on-device array which consists of all local slices stacked
    together in the order consistent with the devices.
  """
  mesh = jax.sharding.Mesh(global_devices, ("devices",))
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("devices"))
  local_ds = mesh.local_devices

  x = np.asarray(memoryview(local_slice))  # No-copy: http://(internal link)
  xs = jax.device_put(np.split(x, len(local_ds), axis=0), local_ds)

  global_shape = (x.shape[0] * jax.process_count(), *x.shape[1:])
  return jax.make_array_from_single_device_arrays(global_shape, sharding, xs)


def get_local_slice_from_fsarray(global_array):
  """Return numpy array for the host-local slice of fully-sharded array.

  Args:
    global_array: JAX array, globally sharded on devices across hosts.

  Returns:
    NumPy array that holds the part of `global_array` that is held by the
    devices on the host that calls this function.
  """
  # For now, for simplicity, we only implement slicing along the first axis.
  for shard in global_array.addressable_shards:
    assert all(idx == slice(None) for idx in shard.index[1:]), (
        f"global_array is sharded along non-first dimensions:\n{shard.index}")

  # Get the shards back in the same order in which the global array was created
  # in the first place. This makes sure it's consistent with other things in the
  # batch, for example (assuming the whole batch is consistent).
  m = {s.device: s for s in global_array.addressable_shards}
  local_shards = [m[d] for d in global_array.sharding.mesh.local_devices]
  return np.concatenate([jax.device_get(s.data) for s in local_shards], axis=0)


def assert_local_slices_same(*global_arrays):
  """Check whether all `global_arrays` have local slices at the same indices."""
  slices = [
      tuple(
          tuple((idx.start, idx.end, idx.step) for idx in s.index)
          for s in a.addressable_shards)
      for a in global_arrays]
  assert len(set(slices)) == 1, f"Not all slices are the same: {slices}"


# TODO: remove this logic when the
# issue is github fixed https://github.com/google/jax/issues/15600.
def jit_cpu(**extra_kwargs):
  def _decorator(fun):
    def _wrapped(*args, **kwargs):
      sh = jax.sharding.SingleDeviceSharding(
          jax.local_devices(backend="cpu")[0]
      )
      return jax.jit(fun, **extra_kwargs, out_shardings=sh)(*args, **kwargs)
    return _wrapped
  return _decorator
