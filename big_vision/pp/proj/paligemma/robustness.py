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

"""pp ops."""

import math

from big_vision.pp import utils
from big_vision.pp.registry import Registry
import tensorflow as tf


@Registry.register("preprocess_ops.resize_r")
@utils.InKeyOutKey()
def get_resize_r(size):
  """Like standard `resize` but randomize some of its parameters."""
  size = utils.maybe_repeat(size, 2)

  # Sadly TF won't let us pass symbolic arguments, so we need to pre-create all
  # variants of function calls we'd like to randomize over...
  resize_fns = [
      lambda x, m=m, a=a: tf.image.resize(x, size, method=m, antialias=a)
      for m in ["bilinear", "bicubic", "lanczos3", "area", "mitchellcubic"]
      for a in [True, False]
  ]

  def _resize_r(image):
    """Resizes image to a given size."""
    dtype = image.dtype
    tf_dtype = tf.type_spec_from_value(image).dtype
    ifn = tf.random.uniform((), 0, len(resize_fns), tf.int32)
    image = tf.switch_case(ifn, [lambda fn=fn: fn(image) for fn in resize_fns])
    return tf.cast(tf.clip_by_value(image, tf_dtype.min, tf_dtype.max), dtype)

  return _resize_r


@Registry.register("preprocess_ops.random_jpeg")
@utils.InKeyOutKey()
def get_random_jpeg(p):
  """With probability `p`, randomly encode-decode as jpeg."""

  fns = [
      lambda x: tf.image.adjust_jpeg_quality(
          x, dct_method="INTEGER_FAST",
          jpeg_quality=tf.random.uniform((), 75, 96, dtype=tf.int32),
      ),
      lambda x: tf.image.adjust_jpeg_quality(
          x, dct_method="INTEGER_ACCURATE",
          jpeg_quality=tf.random.uniform((), 75, 96, dtype=tf.int32),
      ),
  ]

  def _random_jpeg(image):
    """Resizes image to a given size."""
    funcs = [lambda: image] + [lambda fn=fn: fn(image) for fn in fns]
    logits = [math.log(prob) for prob in [1 - p] + [p / len(fns)] * len(fns)]
    fn_idx = tf.random.categorical([logits], 1, dtype=tf.int32)[0, 0]
    return tf.switch_case(fn_idx, funcs)

  return _random_jpeg
