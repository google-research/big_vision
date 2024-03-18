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

"""Tests for builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from big_vision.pp import builder
from big_vision.pp import ops_general  # pylint: disable=unused-import
from big_vision.pp import ops_image  # pylint: disable=unused-import
import numpy as np
import tensorflow.compat.v1 as tf


class BuilderTest(tf.test.TestCase):

  def testSingle(self):
    pp_fn = builder.get_preprocess_fn("resize(256)")
    x = np.random.randint(0, 256, [640, 480, 3])
    image = pp_fn({"image": x})["image"]
    self.assertEqual(image.numpy().shape, (256, 256, 3))

  def testEmpty(self):
    pp_fn = builder.get_preprocess_fn("||inception_crop|||resize(256)||")

    # Typical image input
    x = np.random.randint(0, 256, [640, 480, 3])
    image = pp_fn({"image": x})["image"]
    self.assertEqual(image.numpy().shape, (256, 256, 3))

  def testPreprocessingPipeline(self):
    pp_str = ("inception_crop|resize(256)|resize((256, 256))|"
              "central_crop((80, 120))|flip_lr|value_range(0,1)|"
              "value_range(-1,1)")
    pp_fn = builder.get_preprocess_fn(pp_str)

    # Typical image input
    x = np.random.randint(0, 256, [640, 480, 3])
    image = pp_fn({"image": x})["image"]
    self.assertEqual(image.numpy().shape, (80, 120, 3))
    self.assertLessEqual(np.max(image.numpy()), 1)
    self.assertGreaterEqual(np.min(image.numpy()), -1)

  def testNumArgsException(self):

    x = np.random.randint(0, 256, [640, 480, 3])
    for pp_str in [
        "inception_crop(1)",
        "resize()",
        "resize(1, 1, 1)"
        "flip_lr(1)",
        "central_crop()",
    ]:
      with self.assertRaises(BaseException):
        builder.get_preprocess_fn(pp_str)(x)


if __name__ == "__main__":
  tf.test.main()
