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

"""Transformer encoders for text, similar to CLIP."""

from typing import Any

from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
import flax.linen as nn
import flax.training.checkpoints
import numpy as np

ConfigDict = Any


class _Model(nn.Module):
  """Text transformer similar to CLIP."""

  # Differences to CLIP text encoder (gpt-2) that I am aware of:
  # 1. https://imgur.com/HNi3jix (gpt-1)
  # 2. https://imgur.com/qKGZgBR (gpt-2)
  # 3. https://imgur.com/a/xrpYHF0 (clip)
  # - LayerNorm is on res-path (like pre-activation resnet)
  # - dropout 0.1 everywhere
  # - init as var=0.02, scaled by depth
  # - BOS and EOS tokens, take repr from EOS.
  # - self-attention is autoregressively masked.
  # - scaled in width only, with the image model.

  num_classes: int
  width: int = 512
  depth: int = 12
  mlp_dim: int = 2048
  num_heads: int = 8
  dropout: float = 0.0
  vocab_size: int = 32_000
  pool_type: str = "last"
  scan: bool = False
  remat_policy: str = "nothing_saveable"

  @nn.compact
  def __call__(self, text, *, train=False):
    out = {}

    # We can't use where/argwhere since the output shape is not fixed.
    # Here we use the fact that sequences are padded with EOS tokens, that the
    # EOS token has value 1, and that argmin returns the first index.
    # eos_indices = jnp.argmin(text, axis=1)

    embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.width)
    x = out["embedded"] = embedding(text)

    # Add posemb
    n, l, d = x.shape  # pylint: disable=unused-variable
    x = x + self.param("pos_embedding",
                       nn.initializers.normal(stddev=1/np.sqrt(d)),
                       (1, l, d), x.dtype)

    x, encoder_out = vit.Encoder(
        depth=self.depth, mlp_dim=self.mlp_dim, num_heads=self.num_heads,
        scan=self.scan, remat_policy=self.remat_policy, dropout=self.dropout)(
            x, deterministic=not train)

    out.update({"transformed": x, **encoder_out})

    # Share weights between embeddings and logit transformation.
    out["vocab_logits"] = embedding.attend(x)

    if self.pool_type == "last":
      # Assuming "sticky" EOS tokenization, last token is always EOS.
      x = out["pre_logits"] = x[:, -1, :]
    elif self.pool_type == "first":
      x = out["pre_logits"] = x[:, 0, :]
    elif self.pool_type in ("mean", "gap"):
      x = out["pre_logits"] = x.mean(axis=1)
    elif self.pool_type in ("max", "gmp"):
      x = out["pre_logits"] = x.max(axis=1)
    elif self.pool_type == "map":
      x = out["pre_logits"] = vit.MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    else:
      raise NotImplementedError(f"Cannot do pooling '{self.pool_type}'")

    if self.num_classes:
      x = out["logits"] = nn.Dense(self.num_classes, name="head")(x)
    return x, out


def Model(num_classes, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**vit.decode_variant(variant), **kw})


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  del model_cfg  # unused
  params = utils.load_params(init_file)
  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))

  # Some older (but expensive to train) checkpoints had the posemb added twice
  # by mistake. We detect this here and merge them.
  extra_posemb = params["Encoder_0"].pop("pos_embedding", 0)
  params["pos_embedding"] += extra_posemb

  return common.merge_params(params, init_params, dont_load)
