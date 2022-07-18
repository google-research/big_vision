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

"""Tests for ops_text."""

import big_vision.pp.ops_text as pp
import tensorflow as tf

class PpOpsTest(tf.test.TestCase):
  def test_get_pp_clip_i1k_label_names(self):
    op = pp.get_pp_clip_i1k_label_names()
    labels = op({"label": tf.constant([0, 1])})["labels"].numpy().tolist()
    self.assertAllEqual(labels, ["tench", "goldfish"])

if __name__ == "__main__":
  tf.test.main()
