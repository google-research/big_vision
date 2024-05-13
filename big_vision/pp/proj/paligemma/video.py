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

"""Preprocessing for videos."""

from big_vision.pp import utils
from big_vision.pp.registry import Registry

import tensorflow as tf


@Registry.register('preprocess_ops.video_decode')
def video_decode(res):
  """Preprocessing."""

  def _pp_per_image(img):
    # decode
    return tf.image.resize(tf.io.decode_jpeg(img), (res, res))

  def _pp(data):
    images = data['episodic_images']
    # resize
    images = tf.map_fn(_pp_per_image, images, fn_output_signature=tf.float32)
    # rescale
    images = 2 * (images / 255.) - 1.0
    data['image'] = images
    return data

  return _pp


@Registry.register('preprocess_ops.video_ensure_shape')
def video_ensure_shape(key, shape):
  """Preprocessing."""
  def _video_ensure_shape(data):
    data[key] = tf.ensure_shape(data[key], shape)
    return data

  return _video_ensure_shape


@Registry.register('preprocess_ops.video_replicate_img')
def video_replicate_img(replicas, num_frames):
  """Ensure that for short videos, we have the correct number of frames.

  We replicate and select.

  Args:
    replicas: num_replicas before selection. Should be less than num_frames.
    num_frames: number of frames

  Returns:
    _replicate_img: preprocessing function
  """

  def _replicate_img(data):
    # visual analogies + query
    image = data['image']
    image = tf.tile(image, [replicas, 1, 1, 1])
    data['image'] = image[:num_frames]
    return data

  return _replicate_img


@Registry.register('preprocess_ops.video_choice')
@utils.InKeyOutKey()
def video_choice(empty_fallback=None):
  """Randomly takes one entry out of a tensor after flattening."""

  def _choice(x):
    x = tf.reshape(x, (-1,))  # Ensure it's a 1D array

    # Append the fallback value so we gracefully handle empty cases.
    x0 = tf.zeros(1, x.dtype) if empty_fallback is None else [empty_fallback]
    x = tf.concat([x, x0], axis=0)

    num_choices = tf.maximum(tf.shape(x)[0] - 1, 1)  # Don't sample x0.
    return x[tf.random.uniform([], 0, num_choices, dtype=tf.int32)]

  return _choice


@Registry.register('preprocess_ops.stack_images')
def stack_images(inkeys=(), outkey='image'):

  def _pp(data):
    images = tf.stack([data[inkey] for inkey in inkeys])
    data[outkey] = images
    return data

  return _pp
