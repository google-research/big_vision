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

"""Preprocessing builder."""

from absl import logging
from big_vision.pp.registry import Registry


def get_preprocess_fn(pp_pipeline, log_data=True):
  """Transform an input string into the preprocessing function.

  The minilanguage is as follows:

    fn1|fn2(arg, arg2,...)|...

  And describes the successive application of the various `fn`s to the input,
  where each function can optionally have one or more arguments, which are
  either positional or key/value, as dictated by the `fn`.

  The output preprocessing function expects a dictionary as input. This
  dictionary should have a key "image" that corresponds to a 3D tensor
  (height x width x channel).

  Args:
    pp_pipeline: A string describing the pre-processing pipeline. If empty or
      None, no preprocessing will be executed.
    log_data: Whether to log the data before and after preprocessing. Can also
      be a string to show in the log for debugging, for example dataset name.

  Returns:
    preprocessing function.

  Raises:
    ValueError: if preprocessing function name is unknown
  """

  ops = []
  if pp_pipeline:
    for fn_name in pp_pipeline.split("|"):
      if not fn_name: continue  # Skip empty section instead of error.
      try:
        ops.append(Registry.lookup(f"preprocess_ops.{fn_name}")())
      except SyntaxError as err:
        raise ValueError(f"Syntax error on: {fn_name}") from err

  def _preprocess_fn(data):
    """The preprocessing function that is returned."""
    nonlocal log_data

    # Apply all the individual steps in sequence.
    if log_data:
      logging.info("Data before pre-processing (%s):\n%s", log_data, data)
    for op in ops:
      data = op(data)

    # Validate input
    if not isinstance(data, dict):
      raise ValueError("Argument `data` must be a dictionary, "
                       "not %s" % str(type(data)))

    if log_data:
      logging.info("Data after pre-processing (%s):\n%s", log_data, data)
    log_data = False  # For eager&pygrain: only log first one of each pipeline.
    return data

  return _preprocess_fn
