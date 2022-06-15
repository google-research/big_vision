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

"""Image preprocessing library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from big_vision.pp.registry import Registry
import tensorflow.compat.v1 as tf

TPU_SUPPORTED_DTYPES = [
    tf.bool, tf.int32, tf.int64, tf.bfloat16, tf.float32, tf.complex64,
    tf.uint8, tf.uint32
]


def _remove_tpu_dtypes(data):
  """Recursively removes values with dtype not in TPU_SUPPORTED_DTYPES."""

  def permitted(k, v):
    if v.dtype in TPU_SUPPORTED_DTYPES:
      return True
    tf.logging.warning(
        "Removing key '%s' from data dict because its dtype %s is not in the "
        "supported dtypes: %s", k, v.dtype, TPU_SUPPORTED_DTYPES)
    return False

  return {
      k: _remove_tpu_dtypes(v) if isinstance(v, dict) else v
      for k, v in data.items()
      if isinstance(v, dict) or permitted(k, v)
  }


def get_preprocess_fn(pp_pipeline, remove_tpu_dtypes=True, log_data=True):
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
      None, no preprocessing will be executed, but removing unsupported TPU
      dtypes will still be called if `remove_tpu_dtypes` is True.
    remove_tpu_dtypes: Whether to remove TPU incompatible types of data.
    log_data: Whether to log the data before and after preprocessing. Note:
      Remember set to `False` in eager mode to avoid too many log messages.

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

    # Apply all the individual steps in sequence.
    if log_data:
      logging.info("Data before pre-processing:\n%s", data)
    for op in ops:
      data = op(data)

    # Validate input
    if not isinstance(data, dict):
      raise ValueError("Argument `data` must be a dictionary, "
                       "not %s" % str(type(data)))

    if remove_tpu_dtypes:
      # Remove data that are TPU-incompatible (e.g. filename of type tf.string).
      data = _remove_tpu_dtypes(data)
    if log_data:
      logging.info("Data after pre-processing:\n%s", data)
    return data

  return _preprocess_fn
