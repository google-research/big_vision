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

  def test_setdefault(self):
    data = {
        "empty_image": tf.zeros([0, 0, 0]),
        "image": tf.constant(np.arange(9).reshape(3, 3)),
        "empty_text": tf.zeros([0], tf.string),
        "text": tf.constant(["Hello", "World"], tf.string),
    }
    for out in self.tfrun(pp.get_setdefault("empty_image", 1), data):
      self.assertAllEqual(out["empty_image"], np.array([[[1]]]))
    for out in self.tfrun(pp.get_setdefault("image", 1), data):
      self.assertAllEqual(out["image"], data["image"])
    for out in self.tfrun(pp.get_setdefault("empty_text", "Lucas"), data):
      self.assertAllEqual(out["empty_text"], np.array(["Lucas"]))
    for out in self.tfrun(pp.get_setdefault("text", "Lucas"), data):
      self.assertAllEqual(out["text"], data["text"])

  def _data_for_choice(self):
    return {
        "one_f32": tf.constant([0.42], tf.float32),
        "two_f32": tf.constant([3.14, 0.42], tf.float32),
        "one_str": tf.constant(["Hi"], tf.string),
        "two_str": tf.constant(["Hi", "Lucas"], tf.string),
        "one_vec": tf.reshape(tf.range(2, dtype=tf.float32), (1, 2)),
        "two_vec": tf.reshape(tf.range(4, dtype=tf.float32), (2, 2)),
    }

  def test_choice(self):
    # Test for the default call (n="single")
    data = self._data_for_choice()
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

  def test_choice_nmax(self):
    # n == nelems should be identity (and keep ordering!)
    data = self._data_for_choice()
    for k in ("one_f32", "one_str", "one_vec"):
      for out in self.tfrun(pp.get_choice(n=1, key=[k]), data):
        self.assertAllEqual(out[k], data[k])
      for out in self.tfrun(pp.get_choice(n=[1, 1], key=[k]), data):
        self.assertAllEqual(out[k], data[k])
    for k in ("two_f32", "two_str", "two_vec"):
      for out in self.tfrun(pp.get_choice(n=2, key=[k]), data):
        self.assertAllEqual(out[k], data[k])
      for out in self.tfrun(pp.get_choice(n=[2, 2], key=[k]), data):
        self.assertAllEqual(out[k], data[k])

  def test_choice_n(self):
    # n < nelems should be one of them:
    data = self._data_for_choice()
    for k in ("two_f32", "two_str"):
      for out in self.tfrun(pp.get_choice(n=1, key=[k]), data):
        self.assertIn(out[k], data[k])

    # Special testing for vectors.
    for out in self.tfrun(pp.get_choice(n=1, key=["two_vec"]), data):
      self.assertTrue(tf.logical_or(
          tf.reduce_all(out["two_vec"][0] == data["two_vec"][0]),
          tf.reduce_all(out["two_vec"][0] == data["two_vec"][1]),
      ))

  def test_choice_multi(self):
    # Select consistently across multiple keys.
    data = self._data_for_choice()
    op = pp.get_choice(n=1, key=["two_f32", "two_str"])
    for out in self.tfrun(op, data):
      self.assertTrue(tf.logical_or(
          tf.logical_and(
              tf.reduce_all(out["two_f32"][0] == data["two_f32"][0]),
              tf.reduce_all(out["two_str"][0] == data["two_str"][0]),
          ),
          tf.logical_and(
              tf.reduce_all(out["two_f32"][0] == data["two_f32"][1]),
              tf.reduce_all(out["two_str"][0] == data["two_str"][1]),
          ),
      ))

  def test_choice_n_range(self):
    # n < nelems should be one of them:
    data = self._data_for_choice()
    for k in ("two_f32", "two_str", "two_vec"):
      for out in self.tfrun(pp.get_choice(n=[1, 2], key=[k]), data):
        self.assertTrue(tf.reduce_any([
            tf.reduce_all(out[k] == data[k][0:1]),
            tf.reduce_all(out[k] == data[k][1:2]),
            tf.reduce_all(out[k] == data[k][0:2]),
        ]))


if __name__ == "__main__":
  tf.test.main()
