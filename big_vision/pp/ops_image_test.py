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

"""Tests for ops_image."""

import copy
import io

import big_vision.pp.ops_image as pp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_image_data():
  img = tf.random.uniform((640, 480, 3), 0, 255, tf.int32)  # Can't ask uint8!?
  return {"image": tf.cast(img, tf.uint8)}


class PreprocessOpsTest(tf.test.TestCase):

  def tfrun(self, ppfn, data):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in ppfn(copy.deepcopy(data)).items()}

    # And then once again as part of tfdata pipeline.
    # You'd be surprised how much these two differ!
    tfdata = tf.data.Dataset.from_tensors(copy.deepcopy(data))
    for npdata in tfdata.map(ppfn).as_numpy_iterator():
      yield npdata

  def test_resize(self):
    for data in self.tfrun(pp.get_resize([120, 80]), get_image_data()):
      self.assertEqual(data["image"].shape, (120, 80, 3))

  def test_resize_small(self):
    for data in self.tfrun(pp.get_resize_small(240), get_image_data()):
      self.assertEqual(data["image"].shape, (320, 240, 3))

  def test_resize_long(self):
    for data in self.tfrun(pp.get_resize_long(320), get_image_data()):
      self.assertEqual(data["image"].shape, (320, 240, 3))

  def test_inception_crop(self):
    for data in self.tfrun(pp.get_inception_crop(), get_image_data()):
      self.assertEqual(data["image"].shape[-1], 3)

  def test_decode_jpeg_and_inception_crop(self):
    f = io.BytesIO()
    plt.imsave(f, get_image_data()["image"].numpy(), format="jpg")
    data = {"image": tf.cast(f.getvalue(), tf.string)}
    for data in self.tfrun(pp.get_decode_jpeg_and_inception_crop(), data):
      self.assertEqual(data["image"].shape[-1], 3)

  def test_random_crop(self):
    for data in self.tfrun(pp.get_random_crop([120, 80]), get_image_data()):
      self.assertEqual(data["image"].shape, (120, 80, 3))

  def test_central_crop(self):
    for data in self.tfrun(pp.get_central_crop([20, 80]), get_image_data()):
      self.assertEqual(data["image"].shape, (20, 80, 3))

  def test_random_flip_lr(self):
    data_orig = get_image_data()
    for data in self.tfrun(pp.get_random_flip_lr(), data_orig):
      self.assertTrue(
          np.all(data_orig["image"].numpy() == data["image"]) or
          np.all(data_orig["image"].numpy() == data["image"][:, ::-1]))

if __name__ == "__main__":
  tf.test.main()
