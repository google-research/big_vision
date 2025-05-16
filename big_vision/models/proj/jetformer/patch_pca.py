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

"""Replace VAE with pretrained PCA patch embeddings.

Colab used to train the PCA model:
(internal link)
"""

import functools
import time
from typing import Sequence, Mapping, Any

from absl import logging

from big_vision import utils
from big_vision.models.proj.givt import vae

import einops
import jax
import jax.numpy as jnp


@functools.lru_cache(maxsize=None)
def _load_pca_params(pca_init_file):
  start = time.monotonic()
  x = utils.npload(pca_init_file)
  logging.info("Loaded PCA params %s in %.2f seconds", pca_init_file,
               time.monotonic() - start)
  return x


class Model(vae.Model):
  """Patch PCA embedding model."""

  pca_init_file: str = ""
  noise_std: float = 0.01
  add_dequant_noise: bool = False
  input_size: Sequence[int] = (256, 256)
  patch_size: Sequence[int] = (16, 16)
  whiten: bool = True
  depth_to_seq: int = 1
  skip_pca: bool = False

  def setup(self) -> None:
    assert self.codeword_dim is not None
    assert self.pca_init_file or self.skip_pca
    assert not self.skip_pca or self.depth_to_seq == 1
    if self.skip_pca:
      return

    # Load sklearn PCA model params.
    pca_params = _load_pca_params(self.pca_init_file)
    self.components_ = jnp.asarray(pca_params["components_"], dtype=jnp.float32)
    self.explained_variance_ = jnp.asarray(
        pca_params["explained_variance_"], dtype=jnp.float32)
    self.mean_ = jnp.asarray(pca_params["mean_"], dtype=jnp.float32)

  def _flatten_images(self, patches):
    """Flatten images into a sequence of vectors and apply chroma scaling."""
    flatten = lambda x: einops.rearrange(
        x, "b (h p) (w q) c -> b (h w) (p q c)",
        p=self.patch_size[0], q=self.patch_size[1])
    return flatten(patches)

  def _unflatten_patches(self, patches):
    """Unflatten a sequence of vectors into an image, apply chroma scaling."""
    (h, w), (p, q) = self.input_size, self.patch_size
    unflatten = lambda x, c: einops.rearrange(
        x, "b (h w) (p q c) -> b (h p) (w q) c",
        h=h // p, w=w // q, p=p, q=q, c=c)
    return unflatten(patches, 3)

  def encode(
      self,
      x: jax.Array,
      *,
      train: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    if self.add_dequant_noise:
      x += jax.random.uniform(
          self.make_rng("dropout"), x.shape,
          minval=0.0, maxval=1.0 / 127.5)

    x = self._flatten_images(x)

    if self.skip_pca:
      return x, jnp.zeros_like(x)

    ### Mimic scipy PCA transform code
    x_emb = x @ self.components_.T
    x_emb -= jnp.reshape(self.mean_, (1, -1)) @ self.components_.T
    if self.whiten:
      scale = jnp.sqrt(self.explained_variance_)
      eps = jnp.finfo(scale.dtype).eps
      scale = jnp.where(scale < eps, eps, scale)
      x_emb /= scale
    ###

    if self.depth_to_seq > 1:
      x_emb = einops.rearrange(
          x_emb, "b s (f d) -> b (f s) d", f=self.depth_to_seq)

    if self.noise_std <= 0.0:
      logvar = jnp.zeros_like(x_emb)
    else:
      logvar = 2.0 * jnp.log(jnp.full(x_emb.shape, self.noise_std))

    return x_emb, logvar

  # VAE-like reparametrization - add noise to PCA latents.
  def reparametrize(
      self,
      mu: jax.Array,
      logvar: jax.Array,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    if self.noise_std <= 0.0:
      return mu
    return super().reparametrize(mu, logvar, rng)

  def decode(
      self,
      x: jax.Array,
      train: bool = False,
  ) -> jax.Array | Mapping[str, jax.Array]:
    del train

    if not self.skip_pca:
      if self.depth_to_seq > 1:
        x = einops.rearrange(
            x, "b (f s) d -> b s (f d)", f=self.depth_to_seq)

      ### Mimic scipy PCA inverse transform code
      if self.whiten:
        scaled_components = (
            jnp.sqrt(self.explained_variance_[:, None]) * self.components_)
        x_rec = x @ scaled_components + self.mean_
      else:
        x_rec = x @ self.components_ + self.mean_
      ###
    else:
      x_rec = x

    x_rec = self._unflatten_patches(x_rec)

    return jnp.clip(x_rec, -1.0, 1.0)


def load(*args: Any) -> Any:
  """Dummy loading function returning an empty params dict."""
  del args
  return {}
