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

"""Tests for the FlexiViT model."""

from absl.testing import absltest
from big_vision.models.proj.flexi import vit
import jax
from jax import config
from jax import numpy as jnp
import numpy as np
import tensorflow as tf

config.update("jax_enable_x64", True)


class PatchEmbTest(absltest.TestCase):

  def _test_patch_emb_resize(self, old_shape, new_shape, n_patches=100):
    # This test verifies that if we resize the input image patch and resample
    # the patch embedding accordingly, the output does not change.
    # NOTE: if the image contains more than one patch, then the embeddings will
    # change due to patch interaction during the resizing.
    patch_shape = old_shape[:-2]
    resized_patch_shape = new_shape[:-2]
    patches = np.random.randn(n_patches, *old_shape[:-1])
    w_emb = jnp.asarray(np.random.randn(*old_shape))

    old_embeddings = jax.lax.conv_general_dilated(
        patches, w_emb, window_strides=patch_shape, padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"), precision="highest")

    patch_resized = tf.image.resize(
        tf.constant(patches), resized_patch_shape, method="bilinear").numpy()
    patch_resized = jnp.asarray(patch_resized).astype(jnp.float64)
    w_emb_resampled = vit.resample_patchemb(w_emb, resized_patch_shape)
    self.assertEqual(w_emb_resampled.shape, new_shape)

    new_embeddings = jax.lax.conv_general_dilated(
        patch_resized, w_emb_resampled, window_strides=resized_patch_shape,
        padding="VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision="highest")

    self.assertEqual(old_embeddings.shape, new_embeddings.shape)
    np.testing.assert_allclose(
        old_embeddings, new_embeddings, rtol=1e-1, atol=1e-4)

  def test_resize_square(self):
    out_channels = 256
    patch_sizes = [48, 40, 30, 24, 20, 16, 15, 12, 10, 8, 6, 5]
    for s in patch_sizes:
      old_shape = (s, s, 3, out_channels)
      for t in patch_sizes:
        new_shape = (t, t, 3, out_channels)
        if s <= t:
          self._test_patch_emb_resize(old_shape, new_shape)

  def test_resize_rectangular(self):
    out_channels = 256
    old_shape = (8, 10, 3, out_channels)
    new_shape = (10, 12, 3, out_channels)
    self._test_patch_emb_resize(old_shape, new_shape)

    old_shape = (8, 6, 3, out_channels)
    new_shape = (9, 15, 3, out_channels)
    self._test_patch_emb_resize(old_shape, new_shape)

    old_shape = (8, 6, 3, out_channels)
    new_shape = (15, 9, 3, out_channels)
    self._test_patch_emb_resize(old_shape, new_shape)

  def test_input_channels(self):
    out_channels = 256
    for c in [1, 3, 10]:
      old_shape = (8, 10, c, out_channels)
      new_shape = (10, 12, c, out_channels)
      self._test_patch_emb_resize(old_shape, new_shape)

  def _test_works(self, old_shape, new_shape):
    old = jnp.asarray(np.random.randn(*old_shape))
    resampled = vit.resample_patchemb(old, new_shape[:2])
    self.assertEqual(resampled.shape, new_shape)
    self.assertEqual(resampled.dtype, old.dtype)

  def test_downsampling(self):
    # NOTE: for downsampling we cannot guarantee that the outputs would match
    # before and after downsampling. So, we simply test that the code runs and
    # produces an output of the correct shape and type.
    out_channels = 256
    for t in [4, 5, 6, 7]:
      for c in [1, 3, 5]:
        old_shape = (8, 8, c, out_channels)
        new_shape = (t, t, c, out_channels)
        self._test_works(old_shape, new_shape)

  def _test_raises(self, old_shape, new_shape):
    old = jnp.asarray(np.random.randn(*old_shape))
    with self.assertRaises(AssertionError):
      vit.resample_patchemb(old, new_shape)

  def test_raises_incorrect_dims(self):
    old_shape = (8, 10, 3, 256)
    new_shape = (10, 12, 1, 256)
    self._test_raises(old_shape, new_shape)

    old_shape = (8, 10, 1, 256)
    new_shape = (10, 12, 3, 256)
    self._test_raises(old_shape, new_shape)

    old_shape = (8, 10, 3, 128)
    new_shape = (10, 12, 3, 256)
    self._test_raises(old_shape, new_shape)


if __name__ == "__main__":
  absltest.main()
