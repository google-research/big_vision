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

"""Preprocessing ops."""
from big_vision.pp import utils
from big_vision.pp.registry import Registry
import numpy as np
import tensorflow as tf


@Registry.register("preprocess_ops.rgb_to_grayscale_to_rgb")
@utils.InKeyOutKey(indefault="image", outdefault="image")
def get_rgb_to_grayscale_to_rgb():
  def _rgb_to_grayscale_to_rgb(image):
    return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return _rgb_to_grayscale_to_rgb


@Registry.register("preprocess_ops.nyu_eval_crop")
def get_nyu_eval_crop():
  """Crops labels and image to valid eval area."""
  # crop_h = slice(45, 471)
  # crop_w = slice(41, 601)
  crop_h_start = 54
  crop_h_size = 426
  crop_w_start = 41
  crop_w_size = 560

  def _pp(data):
    tf.debugging.assert_equal(tf.shape(data["labels"]), (480, 640, 1))
    tf.debugging.assert_equal(tf.shape(data["image"]), (480, 640, 3))
    data["labels"] = tf.slice(data["labels"],
                              [crop_h_start, crop_w_start, 0],
                              [crop_h_size, crop_w_size, -1])
    data["image"] = tf.slice(data["image"],
                             [crop_h_start, crop_w_start, 0],
                             [crop_h_size, crop_w_size, -1])
    return data
  return _pp


@Registry.register("preprocess_ops.nyu_depth")
@utils.InKeyOutKey(indefault="depth", outdefault="labels")
def get_nyu_depth():
  """Preprocesses NYU depth data."""
  def _pp(depth):
    return tf.expand_dims(tf.cast(depth, tf.float32), -1)
  return _pp


@Registry.register("preprocess_ops.coco_panoptic")
def get_coco_panoptic_pp():
  """COCO-panoptic: produces a mask with labels and a mask with instance ids.

  Instance channel will have values between 1 and N, and -1 for non-annotated
  pixels.

  Returns:
    COCO panoptic preprocessign op.
  """
  def _coco_panoptic(data):
    instance_ids = tf.cast(data["panoptic_objects"]["id"], tf.int32)
    instance_labels = tf.cast(data["panoptic_objects"]["label"], tf.int32)

    # Convert image with ids split in 3 channels into a an integer id.
    id_mask = tf.einsum(
        "hwc,c->hw",
        tf.cast(data["panoptic_image"], tf.int32),
        tf.constant([1, 256, 256**2], tf.int32))

    # Broadcast into N boolean masks one per instance_id.
    n_masks = tf.cast(
        id_mask[:, :, None] == instance_ids[None, None, :], tf.int32)

    # Merge into a semantic and an instance id mask.
    # Note: pixels which do not belong to any mask, will have value=-1
    # which creates an empty one_hot masks.
    # Number instances starting at 1 (0 is treated specially by make_canonical).
    instance_idx = tf.range(tf.shape(instance_ids)[-1])
    instances = tf.einsum("hwc,c->hw", n_masks, instance_idx + 1)
    semantics = tf.einsum("hwc,c->hw", n_masks, instance_labels + 1)

    data["instances"] = instances[:, :, None]
    data["semantics"] = semantics[:, :, None]
    return data

  return _coco_panoptic


@Registry.register("preprocess_ops.make_canonical")
@utils.InKeyOutKey(indefault="labels", outdefault="labels")
def get_make_canonical(random=False, main_sort_axis="y"):
  """Makes id mask ordered from left to right based on the center of mass."""
  # By convention, instances are in the last channel.
  def _make_canonical(image):
    """Op."""
    instimg = image[..., -1]

    # Compute binary instance masks. Note, we do not touch 0 and neg. ids.
    ids = tf.unique(tf.reshape(instimg, [-1])).y
    ids = ids[ids > 0]
    n_masks = tf.cast(
        instimg[None, :, :] == ids[:, None, None], tf.int32)

    if not random:
      f = lambda x: tf.reduce_mean(tf.cast(tf.where(x), tf.float32), axis=0)
      centers = tf.map_fn(f, tf.cast(n_masks, tf.int64), dtype=tf.float32)
      centers = tf.reshape(centers, (tf.shape(centers)[0], 2))
      major = {"y": 0, "x": 1}[main_sort_axis]
      perm = tf.argsort(
          centers[:, 1 - major] +
          tf.cast(tf.shape(instimg)[major], tf.float32) * centers[:, major])
      n_masks = tf.gather(n_masks, perm)
    else:
      n_masks = tf.random.shuffle(n_masks)

    idx = tf.range(tf.shape(ids)[0])
    can_mask = tf.einsum("chw,c->hw", n_masks, idx + 2) - 1
    # Now, all 0 and neg. ids have collapsed to -1. Thus, we recover 0 id from
    # the original mask.
    can_mask = tf.where(instimg == 0, 0, can_mask)
    return tf.concat([image[..., :-1], can_mask[..., None]], axis=-1)

  return _make_canonical


@Registry.register("preprocess_ops.inception_box")
def get_inception_box(
    *, area=(0.05, 1.0), aspect=(0.75, 1.33), min_obj_cover=0.0,
    outkey="box", inkey="image"):
  """Creates an inception style bounding box which can be used to crop."""
  def _inception_box(data):
    _, _, box = tf.image.sample_distorted_bounding_box(
        tf.shape(data[inkey]),
        area_range=area,
        aspect_ratio_range=aspect,
        min_object_covered=min_obj_cover,
        bounding_boxes=(data["objects"]["bbox"][None, :, :]
                        if min_obj_cover else tf.zeros([0, 0, 4])),
        use_image_if_no_bounding_boxes=True)
    # bbox is [[[y0,x0,y1,x1]]]
    data[outkey] = (box[0, 0, :2], box[0, 0, 2:] - box[0, 0, :2])
    return data
  return _inception_box


@Registry.register("preprocess_ops.crop_box")
@utils.InKeyOutKey(with_data=True)
def get_crop_box(*, boxkey="box"):
  """Crops an image according to bounding box in `boxkey`."""
  def _crop_box(image, data):
    shape = tf.shape(image)[:-1]
    begin, size = data[boxkey]
    begin = tf.cast(begin * tf.cast(shape, tf.float32), tf.int32)
    size = tf.cast(size * tf.cast(shape, tf.float32), tf.int32)
    begin = tf.concat([begin, tf.constant((0,))], axis=0)
    size = tf.concat([size, tf.constant((-1,))], axis=0)
    crop = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    return crop
  return _crop_box


@Registry.register("preprocess_ops.randu")
def get_randu(key):
  """Creates a random uniform float [0, 1) in `key`."""
  def _randu(data):
    data[key] = tf.random.uniform([])
    return data
  return _randu


@Registry.register("preprocess_ops.det_fliplr")
@utils.InKeyOutKey(with_data=True)
def get_det_fliplr(*, randkey="fliplr"):
  """Flips an image horizontally based on `randkey`."""
  # NOTE: we could unify this with regular flip when randkey=None.
  def _det_fliplr(orig_image, data):
    flip_image = tf.image.flip_left_right(orig_image)
    flip = tf.cast(data[randkey] > 0.5, orig_image.dtype)
    return flip_image * flip + orig_image * (1 - flip)
  return _det_fliplr


@Registry.register("preprocess_ops.strong_hash")
@utils.InKeyOutKey(indefault="tfds_id", outdefault="tfds_id")
def get_strong_hash():
  """Preprocessing that hashes a string."""
  def _strong_hash(string):
    return tf.strings.to_hash_bucket_strong(
        string,
        np.iinfo(int).max, [3714561454027272724, 8800639020734831960])
  return _strong_hash
