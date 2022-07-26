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

"""Script that loads a model and only runs evaluators."""

from functools import partial
import importlib

import os

from absl import app
from absl import flags
from absl import logging
import big_vision.evaluators.common as eval_common
import big_vision.utils as u
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
import jax
import jax.config
import jax.numpy as jnp
from ml_collections import config_flags
import tensorflow.io.gfile as gfile


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def main(argv):
  del argv

  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info("Workdir: %s", workdir)

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules", ["ops_general", "ops_image"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # These functions do more stuff internally, for OSS release we mock them by
  # trivial alternatives in order to minize disruptions in the code.
  xid, wid = -1, -1
  def write_note(note):
    if jax.process_index() == 0:
      logging.info("NOTE: %s", note)

  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

  write_note(f"Initializing {config.model_name} model...")
  assert config.get("model.reinit") is None, (
      "I don't think you want any part of the model to be re-initialized.")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  model_kw = dict(config.get("model", {}))
  if "num_classes" in config:  # Make it work for regular + image_text.
    model_kw["num_classes"] = config.num_classes
  model = model_mod.Model(**model_kw)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend="cpu")
  def init(rng):
    input_shapes = config.get("init_shapes", [(1, 224, 224, 3)])
    input_types = config.get("init_types", [jnp.float32] * len(input_shapes))
    dummy_inputs = [jnp.zeros(s, t) for s, t in zip(input_shapes, input_types)]
    return flax.core.unfreeze(model.init(rng, *dummy_inputs))["params"]

  params_cpu = init(jax.random.PRNGKey(42))
  if jax.process_index() == 0:
    parameter_overview.log_parameter_overview(params_cpu, msg="init params")

  # We always load a model, and we don't support checkpointing.
  write_note(f"Initialize model from {config.model_init}...")
  params_cpu = model_mod.load(
      params_cpu, config.model_init, config.get("model"),
      **config.get("model_load", {}))
  if jax.process_index() == 0:
    parameter_overview.log_parameter_overview(params_cpu, msg="restored params")
    num_params = sum(p.size for p in jax.tree_leaves(params_cpu))
    mw.measure("num_params", num_params)

  write_note("Replicating...")
  params_repl = flax_utils.replicate(params_cpu)

  def predict_fn(params, *a, **kw):
    return model.apply({"params": params}, *a, **kw)

  evaluators = eval_common.from_config(
      config, {"predict": predict_fn, "model": model},
      lambda s: write_note(f"Initializing evaluator: {s}..."))

  mw.step_start(0)
  for (name, evaluator, _, prefix) in evaluators:
    write_note(f"{name} evaluation...")
    with u.profile(name):
      for key, value in evaluator.run(params_repl):
        mw.measure(f"{prefix}{key}", value)
  mw.step_end()

  write_note("Done!")
  mw.close()

  # Make sure all hosts stay up until the end of main.
  u.sync_all_hosts()

  if workdir and flags.FLAGS.cleanup and jax.process_index() == 0:
    gfile.rmtree(workdir)
    try:  # Only need this on the last work-unit, if already empty.
      gfile.remove(os.path.join(workdir, ".."))
    except tf.errors.OpError:
      pass


if __name__ == "__main__":
  app.run(main)
