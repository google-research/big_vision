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

"""GIVT-specific preprocessing ops."""

from big_vision.pp import registry
from big_vision.pp import utils
import tensorflow as tf


@registry.Registry.register("preprocess_ops.bin_nyu_depth")
@utils.InKeyOutKey(indefault="labels", outdefault="labels")
def get_bin_nyu_depth(min_depth=0.001, max_depth=10.0, num_bins=256):
  """Binning of NYU depth for UViM in preprocessing rather than model."""

  def _bin_depth(labels):  # pylint: disable=missing-docstring
    labels = (labels - min_depth) / (max_depth - min_depth)
    labels *= num_bins
    labels = tf.cast(tf.floor(labels), tf.int32)
    labels = tf.minimum(labels, num_bins - 1)
    labels = tf.maximum(labels, 0)
    return labels

  return _bin_depth

