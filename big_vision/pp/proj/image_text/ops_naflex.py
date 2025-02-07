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

"""NaFlex (NaViT + FlexiViT) preprocessing ops."""

from big_vision.pp import utils
from big_vision.pp.registry import Registry
import big_vision.utils as u
import tensorflow as tf


def _get_image_size_for_seq(
    image_hw,
    patch_size: int,
    max_sequence_len: int,
    divisible_by_patch: bool = True,
    eps: float = 1e-5):
  """Determine scaling ratio and image size for `get_resize_to_sequence`.

  Did not prove monotonicity necessary for binary search correctness, but this
  works well in practice.

  Args:
    image_hw: Image height and width.
    patch_size: Patchification patch size.
    max_sequence_len: Maximum allowed sequence length for the resulting image.
    divisible_by_patch: If True, the resulting image height and width must be
      divisible by patch size.
    eps: Small number used for binary search convergence.

  Returns:
    ratio: Scaling ratio to applied to image.
    target_hw: Target image height and width taking into account the scaling
      ratio and the `divisible_by_patch` constraint.
  """
  def search_not_done(lb, rb):
    return (rb - lb) >= eps

  def prepare_target_hw(ratio):
    target_hw = tf.cast(image_hw, tf.float32) * ratio
    if divisible_by_patch:
      # Round to multiple of patch size as we want to avoid dropping patches.
      target_hw = patch_size * tf.math.ceil(target_hw / patch_size)
    # Ensure that the image is at least 1 patch in height / width.
    target_hw = tf.maximum(target_hw, patch_size)
    target_hw = tf.cast(target_hw, tf.int32)
    return target_hw

  def is_feasible(ratio):
    target_hw = prepare_target_hw(ratio)
    num_patches = target_hw / patch_size
    sequence_len = tf.math.reduce_prod(num_patches)
    return sequence_len <= max_sequence_len

  def _search_fn(lb, rb):
    mid = (lb + rb) / 2
    return tf.cond(is_feasible(mid), lambda: (mid, rb), lambda: (lb, mid))

  # Left and right boundaries for the binary search.
  state = (tf.constant(eps / 10.), tf.constant(100.0))
  ratio, _ = tf.while_loop(
      search_not_done, _search_fn, state, parallel_iterations=1)
  tf.assert_greater(
      ratio, eps, message="Binary search failed - image too large?")
  tf.assert_less(
      ratio, 100.0, message="Binary search failed - image too small?")

  return ratio, prepare_target_hw(ratio)


@Registry.register("preprocess_ops.resize_to_sequence")
@utils.InKeyOutKey(indefault="image", outdefault="image")
def get_resize_to_sequence(
    patch_size: int,
    max_sequence_len: int,
    divisible_by_patch: bool = True,
    eps: float = 1e-5):
  """Resizes image if it violates restrictions on sequence/side length.

  This op attempts to resize the image in an AR-preserving manner such that:
  - The sequence length of the resulting image (after patchification) is
    maximized, but <= `max_sequence_len`.

  This op *violates* the AR-preserving property if:
  - Image size resulting from the above procedure is not a multiple of patch
    size. In this case AR is distorted to ensure this condition is satisfied.

  Args:
    patch_size: Patchification patch size.
    max_sequence_len: Maximum allowed sequence length for the resulting image.
    divisible_by_patch: If True, the resulting image height and width must be
      divisible by patch size.
    eps: Small number used for binary search convergence.

  Returns:
    Pre-processing op.
  """
  def _resize_fn(image):
    """Performs binary search to find a feasible image size."""
    image_hw = tf.shape(image)[:2]
    _, target_hw = _get_image_size_for_seq(
        image_hw,
        patch_size,
        max_sequence_len,
        divisible_by_patch=divisible_by_patch,
        eps=eps)

    # Actually resize image.
    image = tf.image.resize(
        image,
        target_hw,
        preserve_aspect_ratio=False,
        antialias=True)
    return tf.ensure_shape(image, [None, None, 3])
  return _resize_fn


@Registry.register("preprocess_ops.central_crop_to_sequence")
@utils.InKeyOutKey(indefault="image", outdefault="image")
def get_central_crop_to_sequence(
    patch_size: int,
    max_sequence_len: int,
    divisible_by_patch: bool = True,
    eps: float = 1e-5):
  """Central crops image such that patch sequence length satisfies constraints.

  Constraints used are the as in `resize_to_sequence`.

  Args:
    patch_size: Patchification patch size.
    max_sequence_len: Maximum allowed sequence length for the resulting image.
    divisible_by_patch: If True, the resulting image height and width must be
      divisible by patch size.
    eps: Small number used for binary search convergence.

  Returns:
    Pre-processing op.
  """
  def _central_crop_fn(image):
    image_hw = tf.shape(image)[:2]
    _, target_hw = _get_image_size_for_seq(
        image_hw,
        patch_size,
        max_sequence_len,
        divisible_by_patch=divisible_by_patch,
        eps=eps)

    tf.assert_greater(
        image_hw + 1, target_hw,
        "For central crop the image must be larger than target HW.")
    offset_hw = (image_hw - target_hw) // 2
    image = image[
        offset_hw[0]:offset_hw[0] + target_hw[0],
        offset_hw[1]:offset_hw[1] + target_hw[1],
        :]
    return tf.ensure_shape(image, [None, None, 3])
  return _central_crop_fn


@Registry.register("preprocess_ops.patchify")
@utils.InKeyOutKey(indefault="image", outdefault="image")
def get_patchify(patch_size):
  """Reshapes image into patches and provides patch coordinates."""
  ph, pw = utils.maybe_repeat(patch_size, 2)

  def _patchify(img):
    patches = tf.image.extract_patches(
        img[None, ...], sizes=[1, ph, pw, 1], strides=[1, ph, pw, 1],
        rates=[1, 1, 1, 1], padding="VALID")[0]
    # Patches is now (nh, nw, ph*pw*3), i.e. contains flattened patches.
    nh, nw, d = tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2]

    # Get two (nh, nw) tensors of y/x indices of the patches.
    gy, gx = tf.meshgrid(tf.range(nh), tf.range(nw), indexing="ij")

    return {
        "patches": tf.reshape(patches, (nh * nw, d)),
        "yidx": tf.reshape(gy, [nh * nw]),
        "xidx": tf.reshape(gx, [nh * nw]),
        "type": tf.fill([nh * nw], 1),
    }
  return _patchify


@Registry.register("preprocess_ops.tuplify")
def get_tuplify(inkeys: list[str], outkey: str):
  """Create a tuple of multiple inputs."""
  def tuplify(data):
    data[outkey] = tuple(u.tree_get(data, k) for k in inkeys)
    return data
  return tuplify
