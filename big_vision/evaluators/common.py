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

"""Utils for evaluators in general."""

import dataclasses
import functools
import importlib
from typing import Any, Callable

import flax
import jax
import numpy as np


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


@functools.partial(jax.pmap, axis_name="batch")
def all_gather(x):
  """Gathers variables across all replicas."""
  return jax.lax.all_gather(x, axis_name="batch")


@functools.partial(jax.pmap, axis_name="procs")
def psum(x):
  """Sums variables across all replicas."""
  return jax.lax.psum(x, axis_name="procs")


def global_sum(things):
  """Sums things that are on the host, across all hosts."""
  if jax.process_count() == 1:  # Avoids polluting donut's memory.
    return things

  # Put the thing on the first device, and zeros on all other (local) devices.
  ndev = jax.local_device_count()
  padrep = lambda x: np.array([x] + [np.zeros_like(x)] * (ndev - 1))

  summed_things = psum(jax.tree_map(padrep, things))
  # Now each device holds the global sum, just grab it from the first one.
  return jax.tree_map(lambda x: np.array(x[0]), summed_things)
