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

"""Utils for evaluators in general."""

import dataclasses
import functools
import importlib
import json
import os
from typing import Any, Callable

from absl import flags
from big_vision import input_pipeline
from big_vision.datasets import core as ds_core
from big_vision.pp import builder as pp_builder
import big_vision.utils as u
import flax
import jax
import numpy as np

from tensorflow.io import gfile


def from_config(config, predict_fns,
                write_note=lambda s: s,
                get_steps=lambda key, cfg: cfg[f"{key}_steps"],
                devices=None):
  """Creates a list of evaluators based on `config`."""
  evaluators = []
  specs = config.get("evals", {})

  for name, cfg in specs.items():
    write_note(name)

    # Pop all generic settings off so we're left with eval's kwargs in the end.
    cfg = cfg.to_dict()
    module = cfg.pop("type", name)
    pred_key = cfg.pop("pred", "predict")
    pred_kw = cfg.pop("pred_kw", None)
    prefix = cfg.pop("prefix", f"{name}/")
    cfg.pop("skip_first", None)
    logsteps = get_steps("log", cfg)
    for typ in ("steps", "epochs", "examples", "percent"):
      cfg.pop(f"log_{typ}", None)

    # Use same batch_size as eval by default, to reduce fragmentation.
    # TODO: eventually remove all the deprecated names...
    cfg["batch_size"] = cfg.get("batch_size") or config.get("batch_size_eval") or config.get("input.batch_size") or config.get("batch_size")  # pylint: disable=line-too-long

    module = importlib.import_module(f"big_vision.evaluators.{module}")

    if devices is not None:
      cfg["devices"] = devices

    api_type = getattr(module, "API", "pmap")
    if api_type == "pmap" and "devices" in cfg:
      raise RuntimeError(
          "You are seemingly using the old pmap-based evaluator, but with "
          "jit-based train loop, see (internal link) for more details.")
    if api_type == "jit" and "devices" not in cfg:
      raise RuntimeError(
          "You are seemingly using new jit-based evaluator, but with "
          "old pmap-based train loop, see (internal link) for more details.")

    try:
      predict_fn = predict_fns[pred_key]
    except KeyError as e:
      raise ValueError(
          f"Unknown predict_fn '{pred_key}'. Available predict_fns are:\n"
          + "\n".join(predict_fns)) from e
    if pred_kw is not None:
      predict_fn = _CacheablePartial(predict_fn, flax.core.freeze(pred_kw))
    evaluator = module.Evaluator(predict_fn, **cfg)
    evaluators.append((name, evaluator, logsteps, prefix))

  return evaluators


@dataclasses.dataclass(frozen=True, eq=True)
class _CacheablePartial:
  """partial(fn, **kwargs) that defines hash and eq - to help with jit caches.

  This is particularly common in evaluators when one has many evaluator
  instances that run on difference slices of data.

  Example:

  ```
    f1 = _CacheablePartial(fn, a=1)
    jax.jit(f1)(...)
    jax.jit(_CacheablePartial(fn, a=1))(...)   # fn won't be retraced.
    del f1
    jax.jit(_CacheablePartial(fn, a=1))(...)   # fn will be retraced.
  ```
  """
  fn: Callable[..., Any]
  kwargs: flax.core.FrozenDict

  def __call__(self, *args, **kwargs):
    return functools.partial(self.fn, **self.kwargs)(*args, **kwargs)


def eval_input_pipeline(
    data, pp_fn, batch_size, devices, keep_on_cpu=(),
    cache="pipeline", prefetch=1, warmup=False,
):
  """Create an input pipeline in the way used by most evaluators.

  Args:
    data: The configuration to create the data source (like for training).
    pp_fn: A string representing the preprocessing to be performed.
    batch_size: The batch size to use.
    devices: The devices that the batches are sharded and pre-fetched onto.
    keep_on_cpu: See input_pipeline.start_global. Entries in the batch that
      should be kept on the CPU, hence could be ragged or of string type.
    cache: One of "none", "pipeline", "raw_data", "final_data". Determines what
      part of the input stream should be cached across evaluator runs. They use
      more and more RAM, but make evals faster, in that order.
      - "none": Entirely re-create and destroy the input pipeline each run.
      - "pipeline": Keep the (tf.data) pipeline object alive across runs.
      - "raw_data": Cache the full raw data before pre-processing.
      - "final_data": Cache the full raw data after pre-processing.
    prefetch: How many batches to fetch ahead.
    warmup: Start fetching the first batch at creation time (right now),
      instead of once the iteration starts.

  Returns:
    A tuple (get_iter, steps), the first element is a function that returns
    the iterator to be used for an evaluation, the second one is how many steps
    should be iterated for doing one evaluation.
  """
  assert (
      cache is None
      or cache.lower() in ("none", "pipeline", "raw_data", "final_data")
  ), f"Unknown value for cache: {cache}"
  data_source = ds_core.get(**data)
  tfdata, steps = input_pipeline.make_for_inference(
      data_source.get_tfdata(ordered=True, allow_cache=cache.lower() != "none"),
      batch_size=batch_size,
      num_ex_per_process=data_source.num_examples_per_process(),
      preprocess_fn=pp_builder.get_preprocess_fn(pp_fn, str(data)),
      cache_final=cache == "raw_data",
      cache_raw=cache == "final_data")
  get_data_iter = lambda: input_pipeline.start_global(
      tfdata, devices, prefetch, keep_on_cpu, warmup)

  # Possibly create one persistent iterator:
  if cache in ("pipeline", "raw_data", "final_data"):
    data_iter = get_data_iter()
    get_data_iter = lambda: data_iter

  return get_data_iter, steps


def process_sum(tree):
  """Sums the pytree across all processes."""
  if jax.process_count() == 1:  # Avoids corner-cases on donuts.
    return tree

  with jax.transfer_guard_device_to_host("allow"):
    gathered = jax.experimental.multihost_utils.process_allgather(tree)
  return jax.tree.map(functools.partial(np.sum, axis=0), gathered)


def resolve_outfile(outfile, split="", **kw):
  if not outfile:
    return None

  # A caveat: when workdir doesn't exist but is in the `outfile`, we should
  # skip. This is common in small runs or runlocal debuggings.
  if "{workdir}" in outfile and not flags.FLAGS.workdir:
    return None

  return outfile.format(
      workdir=flags.FLAGS.workdir,
      split="".join(c if c not in "[]%:" else "_" for c in split),
      step=getattr(u.chrono, "prev_step", None),
      **kw,
  )


def multiprocess_write_json(outfile, jobj):  # jobj = "json object"
  """Write a single json file combining all processes' `jobj`s."""
  if not outfile:
    return

  outfile = resolve_outfile(outfile)
  gfile.makedirs(os.path.dirname(outfile))

  if isinstance(jobj, list):
    combine_fn = list.extend
  elif isinstance(jobj, dict):
    combine_fn = dict.update
  else:
    raise TypeError(f"Can only write list or dict jsons, but got {type(jobj)}")

  # First, each process writes its own file.
  with gfile.GFile(outfile + f".p{jax.process_index()}", "w+") as f:
    f.write(json.dumps(jobj))

  u.sync()  # Wait for all files to be written; `with` above does close/flush.

  # Have process 0 collect, concat, and write final output.
  all_json = type(jobj)()
  if jax.process_index() == 0:
    for pid in range(jax.process_count()):
      with gfile.GFile(outfile + f".p{pid}", "r") as f:
        combine_fn(all_json, json.loads(f.read()))
    with gfile.GFile(outfile, "w+") as f:
      f.write(json.dumps(all_json))

  # Cleanup time
  u.sync()
  gfile.remove(outfile + f".p{jax.process_index()}")

  return all_json
