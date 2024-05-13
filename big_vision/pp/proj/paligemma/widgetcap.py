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

"""Widgetcap pp ops."""

from big_vision.pp.registry import Registry
import tensorflow as tf


@Registry.register("preprocess_ops.draw_bbox")
def get_draw_bbox(image_key="image", bbox_key="bbox"):
  """Draw a single bounding box."""

  def _draw_bbox(data):
    """Draw a single bounding box."""
    image = tf.cast(data[image_key], tf.float32)
    image = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0),
        tf.reshape(data[bbox_key], [1, 1, 4]),
        tf.constant([255, 0, 0], dtype=tf.float32, shape=[1, 3]),
    )
    data[image_key] = tf.squeeze(image)
    return data

  return _draw_bbox
