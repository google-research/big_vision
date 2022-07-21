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

"""Tests for pp_ops."""
import copy

from big_vision.pp.proj.uvim import pp_ops as pp
import numpy as np
import tensorflow as tf


def get_image_data(dtype=tf.uint8):
  img = tf.random.uniform((640, 320, 3), 0, 255, tf.int32)  # Can't ask uint8!?
  return {"image": tf.cast(img, dtype)}


class PreprocessOpsTest(tf.test.TestCase):

  def tfrun(self, ppfn, data={}):  # pylint: disable=dangerous-default-value
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in ppfn(copy.deepcopy(data)).items()}

    if not data:  # tf.data doesn't like completely empty dict...k
      data = {"dummy": 0.0}

    # And then once again as part of tfdata pipeline.
    # You'd be surprised how much these two differ!
    tfdata = tf.data.Dataset.from_tensors(copy.deepcopy(data))
    for npdata in tfdata.map(ppfn).as_numpy_iterator():
      yield npdata

  def test_randu(self):
    for output in self.tfrun(pp.get_randu("flip")):
      self.assertEqual(output["flip"].shape, ())
      self.assertAllGreaterEqual(output["flip"], 0.0)
      self.assertAllLessEqual(output["flip"], 1.0)

  def test_det_flip_lr(self):
    # Test both dtypes to make it can be applied correctly to both.
    for dtype in [tf.uint8, tf.float32]:
      image_data = get_image_data(dtype)
      for out in self.tfrun(pp.get_det_fliplr(randkey="rand"),
                            {"rand": 0.1, **image_data}):
        self.assertTrue(np.all(image_data["image"] == out["image"]))
        self.assertEqual(out["image"].dtype, dtype)
      for out in self.tfrun(pp.get_det_fliplr(randkey="rand"),
                            {"rand": 0.6, **image_data}):
        self.assertTrue(np.all(image_data["image"][:, ::-1, :] == out["image"]))
        self.assertEqual(out["image"].dtype, dtype)

  def test_inception_box(self):
    for out in self.tfrun(pp.get_inception_box(), get_image_data()):
      self.assertEqual(out["box"][0].shape, (2,))
      self.assertEqual(out["box"][1].shape, (2,))

  def test_crop_box(self):
    data = get_image_data()
    data["box"] = (tf.constant([0.5, 0.4]), tf.constant([0.25, 0.3]))
    for out in self.tfrun(pp.get_crop_box(), data):
      self.assertEqual(out["image"].shape, (160, 96, 3))
      self.assertAllEqual(
          data["image"][320:320 + 160, 128:128 + 96],
          out["image"])

  def test_make_canonical(self):
    orig = np.array([
        [1, 0, 3, 3, -1],
        [1, 0, 3, 3, -1],
        [1, 0, 2, 2, 2],
        [1, 0, 0, -1, -1]
    ], np.int32)[:, :, None]
    expected = np.array([
        [2, 0, 1, 1, -1],
        [2, 0, 1, 1, -1],
        [2, 0, 3, 3, 3],
        [2, 0, 0, -1, -1]
    ], np.int32)[:, :, None]
    for out in self.tfrun(pp.get_make_canonical(), {"labels": orig}):
      self.assertTrue(np.all(out["labels"] == expected))

    # Test it only affects last channel.
    for out in self.tfrun(pp.get_make_canonical(),
                          {"labels": tf.tile(orig, (1, 1, 3))}):
      self.assertAllEqual(out["labels"][..., 0], orig[..., 0])
      self.assertAllEqual(out["labels"][..., 1], orig[..., 0])
      self.assertAllEqual(out["labels"][..., 2], expected[..., 0])

  def test_nyu_depth(self):
    image = tf.zeros((5, 7, 3), dtype=tf.uint8)
    depth = tf.zeros((5, 7), dtype=tf.float16)
    data = {
        "image": image,
        "depth": depth
    }
    output = pp.get_nyu_depth()(data)
    self.assertEqual(output["image"].shape, (5, 7, 3))
    self.assertEqual(output["image"].dtype, tf.uint8)
    self.assertEqual(output["labels"].shape, (5, 7, 1))
    self.assertEqual(output["labels"].dtype, tf.float32)

  def test_nyu_eval_crop(self):
    image = tf.zeros((480, 640, 3), dtype=tf.uint8)
    depth = tf.zeros((480, 640), dtype=tf.float16)
    data = {
        "image": image,
        "depth": depth
    }
    data = pp.get_nyu_depth()(data)
    output = pp.get_nyu_eval_crop()(data)
    self.assertEqual(output["image"].shape, (426, 560, 3))
    self.assertEqual(output["image"].dtype, tf.uint8)
    self.assertEqual(output["labels"].shape, (426, 560, 1))
    self.assertEqual(output["labels"].dtype, tf.float32)


if __name__ == "__main__":
  tf.test.main()
