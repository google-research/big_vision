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

"""Tests for preprocessing utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from big_vision.pp import utils
import tensorflow.compat.v1 as tf


class UtilsTest(tf.test.TestCase):

  def test_maybe_repeat(self):
    self.assertEqual((1, 1, 1), utils.maybe_repeat(1, 3))
    self.assertEqual((1, 2), utils.maybe_repeat((1, 2), 2))
    self.assertEqual([1, 2], utils.maybe_repeat([1, 2], 2))

  def test_inkeyoutkey(self):
    @utils.InKeyOutKey()
    def get_pp_fn(shift, scale=0):
      def _pp_fn(x):
        return scale * x + shift
      return _pp_fn

    data = {"k_in": 2, "other": 3}
    ppfn = get_pp_fn(1, 2, inkey="k_in", outkey="k_out")  # pylint: disable=unexpected-keyword-arg
    self.assertEqual({"k_in": 2, "k_out": 5, "other": 3}, ppfn(data))

    data = {"k": 6, "other": 3}
    ppfn = get_pp_fn(1, inkey="k", outkey="k")  # pylint: disable=unexpected-keyword-arg
    self.assertEqual({"k": 1, "other": 3}, ppfn(data))

    data = {"other": 6, "image": 3}
    ppfn = get_pp_fn(5, 2)
    self.assertEqual({"other": 6, "image": 11}, ppfn(data))


if __name__ == "__main__":
  tf.test.main()
