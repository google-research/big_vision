# Copyright 2023 Big Vision Authors.
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

"""Tests for ops_general."""

import copy

import big_vision.pp.ops_general as pp
import numpy as np
import tensorflow as tf


class PreprocessOpsTest(tf.test.TestCase):

  def tfrun(self, ppfn, data):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in ppfn(copy.deepcopy(data)).items()}

    # And then once again as part of tfdata pipeline.
    # You'd be surprised how much these two differ!
    tfdata = tf.data.Dataset.from_tensors(copy.deepcopy(data))
    for npdata in tfdata.map(ppfn).as_numpy_iterator():
      yield npdata

  def test_value_range(self):
    img = tf.random.uniform((640, 480, 3), 0, 255, tf.int32)
    data = {"image": tf.cast(img, tf.uint8)}
    for out in self.tfrun(pp.get_value_range(-0.5, 0.5), data):
      self.assertLessEqual(np.max(out["image"]), 0.5)
      self.assertGreaterEqual(np.min(out["image"]), -0.5)

  def test_value_range_custom_input_range(self):
    img = tf.random.uniform((640, 480, 3), 0, 255, tf.int32)
    data = {"image": tf.cast(img, tf.uint8)}
    for out in self.tfrun(pp.get_value_range(-0.5, 0.5, -256, 255, True), data):
      self.assertLessEqual(np.max(out["image"]), 0.5)
      self.assertGreaterEqual(np.min(out["image"]), 0.0)

  def test_get_keep_drop(self):
    data = {"image": 1, "labels": 2, "something": 3}

    for data_keep in self.tfrun(pp.get_keep("image", "labels"), data):
      self.assertAllEqual(set(data_keep.keys()), {"image", "labels"})

    for data_drop in self.tfrun(pp.get_drop("image", "labels"), data):
      self.assertAllEqual(set(data_drop.keys()), {"something"})

  def test_onehot(self):
    data = {"labels": tf.constant(2, dtype=tf.int64)}
    for out in self.tfrun(pp.get_onehot(4, "labels", multi=True), data):
      self.assertAllClose(out["labels"], [0., 0., 1., 0.])

  def test_onehot_multi(self):
    data = {"labels": tf.constant([2, 3, 0], dtype=tf.int64)}
    for out in self.tfrun(pp.get_onehot(4, "labels", multi=False), data):
      self.assertAllClose(out["labels"], [
          [0., 0., 1., 0.],
          [0., 0., 0., 1.],
          [1., 0., 0., 0.]])

    for out in self.tfrun(pp.get_onehot(4, "labels", multi=True), data):
      self.assertAllClose(out["labels"], [1., 0., 1., 1.])

  def test_onehot_2d(self):
    data = {"labels": tf.constant([[2, 3], [0, 1]], dtype=tf.int64)}
    for out in self.tfrun(pp.get_onehot(4, "labels", multi=False), data):
      self.assertAllClose(out["labels"], [
          [[0., 0., 1., 0.], [0., 0., 0., 1.]],
          [[1., 0., 0., 0.], [0., 1., 0., 0.]]])

  def test_onehot_smoothing(self):
    data = {"labels": tf.constant([2, 3, 0], dtype=tf.int64)}
    for out in self.tfrun(
        pp.get_onehot(4, "labels", multi=False, on=0.8, off=0.1), data):
      self.assertAllClose(out["labels"], [
          [0.1, 0.1, 0.8, 0.1],
          [0.1, 0.1, 0.1, 0.8],
          [0.8, 0.1, 0.1, 0.1]])

    for out in self.tfrun(
        pp.get_onehot(4, "labels", multi=True, on=0.8, off=0.1), data):
      self.assertAllClose(out["labels"], [0.8, 0.1, 0.8, 0.8])

  def test_squeeze_last_dim(self):
    data = {"image": tf.constant(np.zeros((32, 32, 3, 1)))}
    for out in self.tfrun(pp.get_squeeze_last_dim(), data):
      self.assertAllEqual(out["image"].shape, [32, 32, 3])

  def test_pad_to_shape(self):
    desired_shape = (8, 10)
    for input_shape in [(8, 4), (8, 3), (8, 10), (8, 1)]:
      data = {"x": tf.ones(input_shape, dtype=tf.float32)}
      for out in self.tfrun(
          pp.get_pad_to_shape(desired_shape, pad_value=-1, key="x"), data):
        self.assertEqual(
            tf.reduce_sum(out["x"]),
            2 * np.prod(input_shape) - np.prod(desired_shape))

  def test_pad_to_shape_none(self):
    data = {"x": tf.ones((8, 4), dtype=tf.float32)}
    for out in self.tfrun(
        pp.get_pad_to_shape((None, 6), pad_value=-1, key="x"), data):
      self.assertEqual(out["x"].shape, (8, 6))
      self.assertEqual(tf.reduce_sum(out["x"]), 8*4 - 8*2)

  def test_pad_to_shape_which_side(self):
    data = {"x": tf.ones((8, 4), dtype=tf.float32)}
    for where, idxs in [("before", [0]), ("both", [0, -1]), ("after", [-1])]:
      for out in self.tfrun(
          pp.get_pad_to_shape((8, 6), key="x", where=where), data):
        self.assertEqual(out["x"].shape, (8, 6))
        self.assertEqual(tf.reduce_sum(out["x"]), 8*4)
        for i in idxs:
          self.assertEqual(out["x"][0, i], 0)

  def test_flatten(self):
    d = {"a": {"b": tf.constant([1, 2, 3])}, "c": "str"}
    self.assertEqual(pp.get_flatten()(d), {
        "a/b": tf.constant([1, 2, 3]),
        "c": "str"
    })

  def test_reshape(self):
    data = {"image": tf.constant(np.zeros((8, 32 * 32 * 3)))}
    for out in self.tfrun(pp.get_reshape(new_shape=(8, 32, 32, 3)), data):
      self.assertAllEqual(out["image"].shape, [8, 32, 32, 3])

  def test_choice(self):
    data = {
        "empty_f32": tf.constant([], tf.float32),
        "one_f32": tf.constant([0.42], tf.float32),
        "two_f32": tf.constant([3.14, 0.42], tf.float32),
        "empty_str": tf.constant([], tf.string),
        "one_str": tf.constant(["Hi"], tf.string),
        "two_str": tf.constant(["Hi", "Lucas"], tf.string),
    }
    self.assertEqual(
        pp.get_choice(inkey="empty_f32", outkey="choice")(data)["choice"], 0.0)
    self.assertEqual(
        pp.get_choice(inkey="empty_str", outkey="choice")(data)["choice"], "")
    self.assertEqual(
        pp.get_choice(inkey="one_f32", outkey="choice")(data)["choice"], 0.42)
    self.assertEqual(
        pp.get_choice(inkey="one_str", outkey="choice")(data)["choice"], "Hi")
    self.assertIn(
        pp.get_choice(inkey="two_f32", outkey="choice")(data)["choice"],
        [3.14, 0.42])
    self.assertIn(
        pp.get_choice(inkey="two_str", outkey="choice")(data)["choice"],
        ["Hi", "Lucas"])

if __name__ == "__main__":
  tf.test.main()
