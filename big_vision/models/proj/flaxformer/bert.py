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

"""BERT encoder, optionally loading pre-trained checkpoints."""

import dataclasses
from typing import Optional

from absl import logging
from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import jax.numpy as jnp
from tensorflow.io import gfile

from flaxformer.architectures.bert import bert
from flaxformer.architectures.bert import bert_checkpoint_converter
from flaxformer.architectures.bert import configs


class Model(nn.Module):
  """BERT encoder with linear projection on last layer CLS token."""

  config: str
  num_classes: Optional[int] = None
  head_zeroinit: bool = True

  @nn.compact
  def __call__(self, text, *, train=False):
    out = {}

    batch_size, max_len = text.shape
    bert_model = bert.BertEncoder(**dataclasses.asdict({
        "base": configs.BertBaseConfig(),
        "large": configs.BertLargeConfig(),
    }[self.config]))
    x = out["transformed"] = bert_model(
        token_ids=text,
        position_ids=jnp.tile(
            jnp.arange(0, max_len, dtype=jnp.int32), [batch_size, 1]),
        segment_ids=jnp.zeros([batch_size, max_len], dtype=jnp.int32),
        input_mask=text.astype(jnp.bool_).astype(jnp.int32),
        enable_dropout=train,
    )

    x = out["pre_logits"] = x[:, 0]  # CLS token

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      x = out["logits"] = nn.Dense(self.num_classes, name="head", **kw)(x)

    return x, out


def load(params, path, model_cfg=None, dont_load=()):
  """Returns `params` with BERT weights replaced from checkpoint at `path`."""
  del model_cfg

  checkpoint_path = f"{path}/bert_model.ckpt"
  if gfile.exists(f"{checkpoint_path}.index"):
    logging.info("Loading original BERT checkpoint from '%s'", checkpoint_path)
    params = flax.core.FrozenDict(params).unfreeze()  # Recursive copy.
    max_len = (
        params["BertEncoder_0"]["embedder"]["embedders_position_ids"]
        ["embedding"].shape[0])
    bert_params, pooler_params = (
        bert_checkpoint_converter.load_params_from_tf_checkpoint(
            checkpoint_path=f"{path}/bert_model.ckpt"))
    del pooler_params
    if isinstance(bert_params, flax.core.FrozenDict):
      bert_params = bert_params.unfreeze()
    bert_params["embedder"]["embedders_position_ids"]["embedding"] = (
        bert_params["embedder"]["embedders_position_ids"]["embedding"][:max_len]
    )
    return common.merge_params(
        {"BertEncoder_0": bert_params}, params, dont_load)

  logging.info(
      "Could not find original BERT checkpoint path '%s', "
      "loading big_vision checkpoint '%s'", checkpoint_path, path)
  restored_params = utils.load_params(path)
  return common.merge_params(restored_params, params, dont_load)
