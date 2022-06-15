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

import importlib


def from_config(config, predict_fns, write_note=lambda s: s):
  """Creates a list of evaluators based on `config`."""
  evaluators = []
  specs = config.get("evals", {})

  for name, cfg in specs.items():
    write_note(name)

    # Pop all generic settings off so we're left with eval's kwargs in the end.
    cfg = cfg.to_dict()
    module = cfg.pop("type", name)
    fn_key = cfg.pop("pred", "predict")
    logsteps = cfg.pop("log_steps", None)
    prefix = cfg.pop("prefix", f"{name}/")

    # Use same batch_size as eval by default, to reduce fragmentation.
    cfg["batch_size"] = cfg.get("batch_size") or config.get("batch_size_eval") or config.get("batch_size")  # pylint: disable=line-too-long

    module = importlib.import_module(f"big_vision.evaluators.{module}")
    evaluator = module.Evaluator(predict_fns[fn_key], **cfg)
    evaluators.append((name, evaluator, logsteps, prefix))

  return evaluators
