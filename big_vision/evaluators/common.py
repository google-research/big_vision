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


def from_config(config, model, default_batch_size=None, write_note=lambda s: s):
  """Creates a list of evaluators based on `config`."""
  evaluators = []
  specs = config.get("evals", [])

  for spec in specs:
    if isinstance(spec, str):  # A shortcut when name == module
      spec = (spec, spec)
    name, module = spec
    write_note(name)

    # Sanitize the config.
    cfg = config.get(name).to_dict() if name in config else {}
    # Use same batch_size as eval by default, to reduce fragmentation.
    cfg["batch_size"] = cfg.get("batch_size") or default_batch_size
    logsteps, prefix = cfg.pop("log_steps", None), cfg.pop("prefix", f"{name}/")
    module = importlib.import_module(f"big_vision.evaluators.{module}")
    evaluator = module.Evaluator(model, **cfg)
    evaluators.append((name, evaluator, logsteps, prefix))

  return evaluators
