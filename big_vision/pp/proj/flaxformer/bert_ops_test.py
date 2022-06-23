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

"""Tests for bert_ops."""

import tempfile
from unittest import mock

from big_vision import input_pipeline
import big_vision.pp.builder as pp_builder
import big_vision.pp.ops_general  # pylint: disable=unused-import
from big_vision.pp.proj.flaxformer import bert_ops  # pylint: disable=unused-import
import tensorflow as tf


# BERT vocabulary for testing.
_BERT_VOCAB = [
    "[PAD]",
    "[UNK]",
    "more",
    "than",
    "one",
    "[CLS]",
    "[SEP]",
]


def _create_ds(
    mock_builder, pp_str, tensor_slices, num_examples, remove_tpu_dtypes):
  ds = tf.data.Dataset.from_tensor_slices(tensor_slices)
  mock_builder.return_value.as_dataset.return_value = ds
  mock_builder.return_value.info.splits.__getitem__.return_value.num_examples = num_examples
  return input_pipeline.make_for_inference(
      dataset="mocked",
      split="test",
      preprocess_fn=pp_builder.get_preprocess_fn(
          pp_str, remove_tpu_dtypes=remove_tpu_dtypes),
      batch_size=num_examples,
  )[0]


class BertOpsTest(tf.test.TestCase):

  @mock.patch("tensorflow_datasets.builder")
  def test_tokenize(self, mock_builder):
    inkey = "texts"
    vocab_path = f"{tempfile.mkdtemp()}/vocab.txt"
    with open(vocab_path, "w") as f:
      f.write("\n".join(_BERT_VOCAB))
    pp_str = (
        f"bert_tokenize(inkey='{inkey}', vocab_path='{vocab_path}', "
        f"max_len=5)"
    )
    tensor_slices = {
        inkey: tf.ragged.constant([["one more"], ["more than one"], [""]])
    }
    ds = _create_ds(mock_builder, pp_str, tensor_slices, 3, True)
    self.assertAllEqual(
        next(iter(ds))["labels"],
        [[5, 4, 2, 0, 0], [5, 2, 3, 4, 0], [5, 0, 0, 0, 0]],
    )


if __name__ == "__main__":
  tf.test.main()
