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

"""Model definition to train a single ViT model with the contrastive trainer."""

import importlib
from typing import Optional, Any

from big_vision import utils
import flax.linen as nn
import jax.numpy as jnp

ConfigDict = Any


class Model(nn.Module):
  """Single ViT to encode regular images and text images."""
  image: Optional[ConfigDict] = None
  image_model: str = "vit"
  out_dim: int = 768
  temperature_init: float = 10.0

  @nn.compact
  def __call__(self, image, text=None, **kw):
    """Returns (B, C) image and (B, C) text representations, and some extras."""
    ztxt, zimg = None, None
    kw = kw or {}

    image_model = importlib.import_module(
        f"big_vision.models.{self.image_model}"
    ).Model(**{"num_classes": self.out_dim, **(self.image or {})}, name="img")  # pylint: disable=not-a-mapping

    def _compute_embedding(input_image, prefix):
      zemb, out_emb = image_model(input_image, **kw)
      out = {f"{prefix}/{k}": v for k, v in out_emb.items()}

      # Normalize the embeddings.
      out[f"{prefix}/norm"] = jnp.linalg.norm(zemb, axis=1, keepdims=True)
      out[f"{prefix}/normalized"] = zemb = zemb / (out[f"{prefix}/norm"] + 1e-8)
      return zemb, out

    out = {}
    if image is not None:
      zimg, out_img = _compute_embedding(image, "img")
      out.update(out_img)

    if text is not None:
      ztxt, out_txt = _compute_embedding(text, "txt")
      out.update(out_txt)

    temp_init = jnp.log(self.temperature_init)
    t = self.param("t",
                   lambda key, shape, dtype: temp_init*jnp.ones(shape, dtype),
                   (1,), jnp.float32)
    out["t"] = jnp.exp(t)
    out["t/parameter"] = t

    return zimg, ztxt, out


def load(init_params, init_files, model_cfg, img_load_kw={}):  # pylint: disable=dangerous-default-value
  """Loads the ViT parameters - adapted from proj/image_text/two_towers.py."""
  if isinstance(init_files, str):
    # A shortcut for a single file checkpoint of a two_towers model.
    init_files = {k: f"{init_files}:{k}" for k in ("img", "t")}
  else:
    init_files = {**init_files}  # Shallow copy because we'll pop stuff off.

  restored_params = {**init_params}

  img_init = init_files.pop("image", init_files.pop("img", None))
  if img_init:
    restored_params["img"] = importlib.import_module(
        f"big_vision.models.{model_cfg.image_model}"
    ).load(init_params["img"], img_init, model_cfg.image, **img_load_kw)

  t_init = init_files.pop("temperature", init_files.pop("t", None))
  if t_init:
    restored_params["t"] = utils.load_params(None, t_init)

  assert not init_files, (
      f"There's something unused left in `config.model_init`. You probably got "
      f"a typo. Here it is: {init_files}")

  return restored_params
