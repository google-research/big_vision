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

"""Tests for NaFlex preprocessing ops."""

import copy

from absl.testing import parameterized
from big_vision.pp.proj.image_text import ops_naflex as pp
import numpy as np
import tensorflow as tf


def get_image_data(h, w):
  img = tf.random.uniform((h, w, 3), 0, 255, tf.int32)  # Can't ask uint8!?
  return {"image": tf.cast(img, tf.uint8)}


class NaflexTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, ppfn, data):
    # Run once as standalone, as could happen eg in colab.
    yield tf.nest.map_structure(np.array, ppfn(copy.deepcopy(data)))

    # And then once again as part of tfdata pipeline.
    # You'd be surprised how much these two differ!
    tfdata = tf.data.Dataset.from_tensors(copy.deepcopy(data))
    for npdata in tfdata.map(ppfn).as_numpy_iterator():
      yield npdata

  @parameterized.parameters(
      (6, 8),
      (7, 9),
      (8, 10),
  )
  def test_patchify_valid(self, h, w):
    """Tests the patchification op."""
    op = pp.get_patchify((3, 4))
    inputs = get_image_data(h, w)
    for data in self.tfrun(op, inputs):
      self.assertEqual(data["image"]["patches"].shape, (4, 3*4*3))
      self.assertAllEqual(
          data["image"]["patches"][-1],
          np.array(inputs["image"])[3:6, 4:8, :].flatten())
      self.assertAllEqual(data["image"]["yidx"], [0, 0, 1, 1])
      self.assertAllEqual(data["image"]["xidx"], [0, 1, 0, 1])

  @parameterized.named_parameters([
      ("square_121_exact", (48, 48), 3, 121, (33, 33)),
      ("square_225_inexact", (112, 109), 7, 225, (105, 105)),
      ("square_64_exact", (176, 176), 11, 64, (88, 88)),
      ("rect_12_exact", (256, 64), 16, 12, (96, 32)),
      ("rect_15_exact_ps8", (256, 64), 8, 15, (56, 16)),
      ("rect_16_inexact", (63, 241), 16, 16, (32, 128)),
      ("rect_less_than_patch", (16, 512), 16, 16, (16, 256)),
  ])
  def test_pp_resize_to_sequence(
      self, image_size, patch_size, seq_len, expected_image_size):
    """Tests the AR-preserving `resize_to_sequence` op."""
    op = pp.get_resize_to_sequence(patch_size, seq_len)
    inputs = get_image_data(*image_size)
    for outputs in self.tfrun(op, inputs):
      self.assertAllEqual(outputs["image"].shape, expected_image_size + (3,))

if __name__ == "__main__":
  tf.test.main()
