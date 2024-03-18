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

"""Tests for bert."""

import tempfile

from big_vision import input_pipeline
from big_vision.models.proj.flaxformer import bert
from big_vision.models.proj.flaxformer import bert_test_util
import big_vision.pp.builder as pp_builder
import big_vision.pp.ops_general  # pylint: disable=unused-import
import big_vision.pp.proj.flaxformer.bert_ops  # pylint: disable=unused-import
import flax
import jax
import jax.numpy as jnp
import tensorflow as tf


# BERT vocabulary for testing.
_BERT_VOCAB = [
    "[PAD]",
    "[UNK]",
    "this",
    "is",
    "a",
    "test",
    "[CLS]",
    "[SEP]",
]
_TOKEN_LEN = 16


class BertTest(tf.test.TestCase):

  def test_load_apply(self):
    inkey = "text"
    vocab_path = f"{tempfile.mkdtemp()}/vocab.txt"
    with open(vocab_path, "w") as f:
      f.write("\n".join(_BERT_VOCAB))
    ds2, _ = input_pipeline.make_for_inference(
        tf.data.Dataset.from_tensor_slices(
            {inkey: tf.ragged.constant([["this is a test"]])}),
        num_ex_per_process=[1],
        preprocess_fn=pp_builder.get_preprocess_fn(
            f"bert_tokenize(inkey='{inkey}', vocab_path='{vocab_path}', "
            f"max_len={_TOKEN_LEN})"
            "|keep('labels')"),
        batch_size=1,
    )
    text = jnp.array(next(iter(ds2))["labels"])
    model = bert.Model(config="base")
    variables = model.init(jax.random.PRNGKey(0), text)
    params = bert.load(flax.core.unfreeze(variables)["params"],
                       bert_test_util.create_base_checkpoint())
    x, out = model.apply({"params": params}, text)
    self.assertAllEqual(jax.tree_map(jnp.shape, x), (1, 768))
    self.assertAllEqual(
        jax.tree_map(jnp.shape, out), {
            "transformed": (1, 16, 768),
            "pre_logits": (1, 768),
        })


if __name__ == "__main__":
  tf.test.main()
