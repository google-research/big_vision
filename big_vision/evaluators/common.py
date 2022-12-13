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

"""Utils for evaluators in general."""

import dataclasses
import functools
import importlib
from typing import Any, Callable

import flax


def from_config(config, predict_fns,
                write_note=lambda s: s,
                get_steps=lambda key, cfg: cfg[f"{key}_steps"]):
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
    logsteps = get_steps("log", cfg)
    for typ in ("steps", "epochs", "examples", "percent"):
      cfg.pop(f"log_{typ}", None)

    # Use same batch_size as eval by default, to reduce fragmentation.
    # TODO: eventually remove all the deprecated names...
    cfg["batch_size"] = cfg.get("batch_size") or config.get("batch_size_eval") or config.get("input.batch_size") or config.get("batch_size")  # pylint: disable=line-too-long

    module = importlib.import_module(f"big_vision.evaluators.{module}")
    predict_fn = predict_fns[pred_key]
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
