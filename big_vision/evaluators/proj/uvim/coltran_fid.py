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

"""Evaluation producing ColTran FID-5K metric."""

import functools
import os

from absl import logging
import einops
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

from tensorflow.io import gfile


ROOT = os.environ.get("FID_DATA_DIR", ".")


def _preprocess(image, resolution=512):
  """ColTran dataset preprocessing.

  See,
  github.com/google-research/google-research/blob/master/coltran/datasets.py#L44

  Args:
    image: ImageNet example from TFDS.
    resolution: Integer representing output size.

  Returns:
    An int32 image of size (resolution, resolution, 3).
  """
  image_shape = tf.shape(image)
  height, width = image_shape[0], image_shape[1]
  side_size = tf.minimum(height, width)
  image = tf.image.resize_with_crop_or_pad(
      image, target_height=side_size, target_width=side_size)
  image = tf.image.resize(image, method="area", antialias=True,
                          size=(resolution, resolution))
  image = tf.cast(tf.round(image), dtype=tf.int32)
  return image


def _normalize(x):
  """Coltran normalization to expected range for Inception module.

  Args:
    x: Image with values in [0,255].

  Returns:
    Image with values in [-1,1].
  """
  x = tf.cast(x, tf.float32)
  x = (x / 128.0) - 1.0  # note: 128.0 is the value used in ColTran.
  return x


class Evaluator:
  """ColTran FID-5K Evaluator.

  This Evaluator aims to mirror the evaluation pipeline used by Kumar et.al.
  in Colorization Transformer (https://arxiv.org/abs/2102.04432).

  To be clear: much of this code is direct snippets from ColTran code.

  See,
  github.com/google-research/google-research/blob/master/coltran/datasets.py#L44

  The ColTran pipeline has numerous stages, where serialied data is passed
  between binaries via file, etc...  While we don't physically write the same
  files, we simulate the effects of the serialization (e.g., quantization).
  """

  def __init__(self,
               predict_fn,
               batch_size,  # ignored
               device_batch_size=5,
               coltran_seed=1,
               predict_kwargs=None):
    """Create Evaluator.

    Args:
      predict_fn: Colorization prediction function.  Expects grayscale images
        of size (512, 512, 3) in keys `image` and `image_ctx` with values in
        the range [-1,1].  Outputs `color` image in range [-1,1].
      batch_size: ignored.
      device_batch_size: number of images per batch, per device.
      coltran_seed: used to specify the block of 5_000 images used to generate
        the reference pool.  Value of `1` matches default ColTran code.
      predict_kwargs: arguments passed to `predict_fn`.
    """
    del batch_size

    self.num_devices = jax.local_device_count()
    self.device_batch_size = device_batch_size
    logging.log(logging.INFO, "Colorizing with batch size %i on %i devices.",
                self.device_batch_size, self.num_devices)
    assert 5_000 % (self.device_batch_size * self.num_devices) == 0

    predict = functools.partial(predict_fn, **(predict_kwargs or {}))
    self.predict_fn = jax.pmap(predict)

    module = tfhub.load(tfgan.eval.INCEPTION_TFHUB)
    def _pools(x):
      return np.squeeze(module(x)[tfgan.eval.INCEPTION_FINAL_POOL].numpy())

    self.inception_pool = _pools

    # Setup the colorization dataset.
    # TRICKY: ColTran FID-5k uses the first 5_000 images returned as read by
    # default from tensorflow_datasets (that is: with shard interleaving).
    # In particular note that it is different than the set of images returned
    # by "validation[:5000]".
    def _eval_data_preprocess(example):
      # Colorization happens at 512x512 resolution.
      image = _preprocess(example["image"], resolution=512)
      image = _normalize(image)
      grayscale = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
      return {
          "image": image,
          "grayscale": grayscale,
          "file_name": example["file_name"]
      }

    ds = tfds.load("imagenet2012", split="validation")
    ds = ds.map(_eval_data_preprocess)
    ds = ds.take(5_000)
    ds = ds.batch(self.device_batch_size)
    ds = ds.batch(self.num_devices)
    self.eval_data = ds.cache().prefetch(tf.data.AUTOTUNE)

    # Setup the reference dataset.
    def _reference_data_preprocess(example):
      # ColTran eval operates on 256x256.
      image = _preprocess(example["image"], resolution=256)
      image = _normalize(image)
      return {"image": image, "file_name": example["file_name"]}

    ds = tfds.load("imagenet2012", split="validation")
    ds = ds.map(_reference_data_preprocess)
    # Skip the images used in colorization.
    ds = ds.skip(5_000)
    # ColTran eval w/ seed=1 effectively uses 10_000:15_000 to
    # calculate reference.
    ds = ds.skip(coltran_seed * 5_000)
    ds = ds.take(5_000)
    ds = ds.batch(device_batch_size)
    self.reference_data = ds.cache().prefetch(tf.data.AUTOTUNE)

    def _get_file(name):
      return os.path.join(ROOT, name)

    with gfile.GFile(_get_file("eval_file_names.txt")) as f:
      self.eval_file_names = frozenset(f.read().splitlines())

    with gfile.GFile(_get_file("reference_file_names.txt")) as f:
      self.reference_file_names = frozenset(f.read().splitlines())

  def run(self, params):
    """Run eval."""

    if jax.process_index():  # Host0 does all work.
      return

    color_pools = []
    color_file_names = set()
    for i, batch in enumerate(self.eval_data.as_numpy_iterator()):
      predict_batch = {
          "labels": batch["image"],
          "image": batch["grayscale"],
          "image_ctx": batch["grayscale"],
      }
      y = self.predict_fn(params, predict_batch)
      y = y["color"]
      y = einops.rearrange(y, "d b h w c -> (d b) h w c")

      # Return to the ColTran eval size of 256x256.
      y = tf.image.resize(y, (256, 256), "area")

      # Mimic effect of serializing image as integers and map back to [-1, 1].
      y = np.clip(np.round((y + 1.) * 128.), 0, 255)
      y = _normalize(y)

      color_pools.append(self.inception_pool(y))

      file_names = einops.rearrange(batch["file_name"], "d b -> (d b)")
      color_file_names.update([f.decode() for f in file_names])

      logging.log_every_n_seconds(
          logging.INFO,
          "ColTran FID eval: processed %i colorized examples so far.", 30,
          (i + 1) * self.device_batch_size * self.num_devices)

    reference_pools = []
    reference_file_names = set()
    for i, batch in enumerate(self.reference_data.as_numpy_iterator()):
      image = batch["image"]
      assert np.array_equal(image.shape, (self.device_batch_size, 256, 256, 3))
      reference_pools.append(self.inception_pool(image))
      reference_file_names.update([f.decode() for f in batch["file_name"]])

      logging.log_every_n_seconds(
          logging.INFO,
          "ColTran FID eval: processed %i reference examples so far.", 30,
          (i + 1) * self.device_batch_size)

    if color_file_names != self.eval_file_names:
      raise ValueError("unknown: {}\nmissing: {}".format(
          color_file_names - self.eval_file_names,
          self.eval_file_names - color_file_names))

    if reference_file_names != self.reference_file_names:
      raise ValueError("unknown: {}\nmissing: {}".format(
          reference_file_names - self.reference_file_names,
          self.reference_file_names - reference_file_names))

    color = np.concatenate(color_pools, axis=0)
    reference = np.concatenate(reference_pools, axis=0)

    if color.shape[0] != 5_000:
      raise ValueError(color.shape)

    if reference.shape[0] != 5_000:
      raise ValueError(reference.shape)

    yield "FID_5k", tfgan.eval.frechet_classifier_distance_from_activations(
        color, reference)
