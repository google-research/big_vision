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

"""Transformer encoders both for text and for images."""

import importlib
from typing import Any, Optional, Tuple, Union
from absl import logging

from big_vision import utils
import flax.linen as nn
import jax.numpy as jnp

ConfigDict = Any


class Model(nn.Module):
  """Two towers transformer."""
  image: Optional[ConfigDict] = None
  text: Optional[ConfigDict] = None
  text_model: str = "proj.image_text.text_transformer"
  image_model: str = "vit"
  out_dim: Union[int, Tuple[int, int]] = 128
  temperature_init: float = 1.0
  bias_init: Optional[float] = None

  @nn.compact
  def __call__(self, image, text=None, **kw):
    """Returns (B,C) image and (B,C) text representations."""

    # Support calling without text or without image, for example for few-shot.
    ztxt, zimg = None, None
    out = {}
    out_dims = self.out_dim
    if isinstance(out_dims, int):
      out_dims = (out_dims, out_dims)

    # Embed the text:
    if text is not None:
      text_model = importlib.import_module(
          f"big_vision.models.{self.text_model}"
      ).Model(**{"num_classes": out_dims[1], **(self.text or {})}, name="txt")

      ztxt, out_txt = text_model(text, **kw)
      for k, v in out_txt.items():
        out[f"txt/{k}"] = v

      # Normalize the embeddings the models give us.
      out["txt/norm"] = jnp.linalg.norm(ztxt, axis=1, keepdims=True)
      out["txt/normalized"] = ztxt = ztxt / (out["txt/norm"] + 1e-8)

    if image is not None:
      image_model = importlib.import_module(
          f"big_vision.models.{self.image_model}"
      ).Model(**{"num_classes": out_dims[0], **(self.image or {})}, name="img")  # pylint: disable=not-a-mapping

      zimg, out_img = image_model(image, **kw)
      for k, v in out_img.items():
        out[f"img/{k}"] = v

      # Normalize the embeddings the models give us.
      out["img/norm"] = jnp.linalg.norm(zimg, axis=1, keepdims=True)
      out["img/normalized"] = zimg = zimg / (out["img/norm"] + 1e-8)

    temp_init = jnp.log(self.temperature_init)
    t = self.param("t",
                   lambda key, shape, dtype: temp_init * jnp.ones(shape, dtype),
                   (1,), jnp.float32)
    out["t"] = jnp.exp(t)

    out["t/parameter"] = t
    if (b_init := self.bias_init) is not None:
      out["b"] = self.param("b", lambda k, s, d: b_init * jnp.ones(s, d),
                            (1,), jnp.float32)

    # We could actually play with pre-multiplying by temperature here, such
    # that out["t"] is nothing special to the trainer anymore.

    return zimg, ztxt, out


def load(init_params, init_files, model_cfg, img_load_kw={}, txt_load_kw={}):  # pylint: disable=dangerous-default-value
  """Loads both towers, `init_files` is now a dict with `img` and `txt` keys."""
  if isinstance(init_files, str):
    init_files = VANITY_NAMES.get(init_files, init_files)

  if isinstance(init_files, str):
    # A shortcut for a single file checkpoint of a two_towers model.
    if "bias_init" in model_cfg.keys():
      logging.info("loading img, txt, t, and b from a single checkpoint.")
      init_files = {k: f"{init_files}:{k}" for k in ("img", "txt", "t", "b")}
    else:
      logging.info("loading img, txt, and t from a single checkpoint.")
      init_files = {k: f"{init_files}:{k}" for k in ("img", "txt", "t")}
  else:
    init_files = {**init_files}  # Shallow copy because we'll pop stuff off.

  if not init_params:  # Convenience to skip checks in colab.
    init_params = {"img": None, "txt": None}
  restored_params = {**init_params}

  img_init = init_files.pop("image", init_files.pop("img", None))
  if img_init:
    restored_params["img"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('image_model', 'vit')}"
    ).load(init_params["img"], img_init, model_cfg.image, **img_load_kw)

  txt_init = init_files.pop("text", init_files.pop("txt", None))
  if txt_init:
    restored_params["txt"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('text_model', 'proj.image_text.text_transformer')}"  # pylint: disable=line-too-long
    ).load(init_params["txt"], txt_init, model_cfg.text, **txt_load_kw)

  t_init = init_files.pop("temperature", init_files.pop("t", None))
  if t_init:
    restored_params["t"] = utils.load_params(t_init)

  b_init = init_files.pop("bias", init_files.pop("b", None))
  if b_init:
    restored_params["b"] = utils.load_params(b_init)

  assert not init_files, (
      f"There's something unused left in `config.model_init`. You probably got "
      f"a typo. Here it is: {init_files}")

  return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # SigLIP image encoder checkpoints from https://arxiv.org/abs/2303.15343
    "SigLIP B/16 224": "gs://big_vision/siglip/webli_en_b16_224_63724782.npz",
    "SigLIP B/16 256": "gs://big_vision/siglip/webli_en_b16_256_60500360.npz",
    "SigLIP B/16 384": "gs://big_vision/siglip/webli_en_b16_384_68578854.npz",
    "SigLIP B/16 512": "gs://big_vision/siglip/webli_en_b16_512_68580893.npz",
    "SigLIP L/16 256": "gs://big_vision/siglip/webli_en_l16_256_60552751.npz",
    "SigLIP L/16 384": "gs://big_vision/siglip/webli_en_l16_384_63634585.npz",
    "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz",
    "SigLIP So400m/14 384": "gs://big_vision/siglip/webli_en_so400m_384_58765454.npz",
    "SigLIP B/16-i18n 256": "gs://big_vision/siglip/webli_i18n_b16_256_66117334.npz",
    # pylint: enable=line-too-long
}
