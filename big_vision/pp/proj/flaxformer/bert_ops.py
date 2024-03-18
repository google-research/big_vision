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

"""BERT-related preprocessing ops (using WordPiece tokenizer)."""

from big_vision.pp import utils
from big_vision.pp.registry import Registry
import tensorflow as tf
import tensorflow_text


# Internally using
# BasicTokenizer
# https://github.com/tensorflow/text/blob/df5250d6cf1069990df4bf55154867391ab5381a/tensorflow_text/python/ops/bert_tokenizer.py#L67
# WordpieceTokenizer
# https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/wordpiece_tokenizer.py
def _create_bert_tokenizer(vocab_path):
  """Returns cls_token id and tokenizer to use in a tf.Dataset.map function."""
  # Create tokenizer inside a tf.init_scope so the vocab is only loaded from
  # disk once per dataset iterator (see: http://(internal link)).
  # TODO: Make a local copy of vocab if creating many iterators.
  with tf.init_scope():
    tokenizer = tensorflow_text.BertTokenizer(
        vocab_path,
        token_out_type=tf.int32,
        lower_case=True,
    )

  with tf.io.gfile.GFile(vocab_path) as f:
    vocab = f.read().split("\n")
  cls_token = vocab.index("[CLS]")

  return cls_token, tokenizer


@Registry.register("preprocess_ops.bert_tokenize")
@utils.InKeyOutKey(indefault=None, outdefault="labels")
def get_pp_bert_tokenize(vocab_path, max_len, sample_if_multi=True):
  """Extracts tokens with tensorflow_text.BertTokenizer.

  Args:
    vocab_path: Path to a file containing the vocabulry for the WordPiece
      tokenizer. It's the "vocab.txt" file in the zip file downloaded from
      the original repo https://github.com/google-research/bert
    max_len: Number of tokens after tokenization.
    sample_if_multi: Whether the first text should be taken (if set to `False`),
      or whether a random text should be tokenized.

  Returns:
    A preprocessing Op.
  """

  cls_token, tokenizer = _create_bert_tokenizer(vocab_path)

  def _pp_bert_tokenize(labels):

    labels = tf.reshape(labels, (-1,))
    labels = tf.concat([labels, [""]], axis=0)
    if sample_if_multi:
      num_texts = tf.maximum(tf.shape(labels)[0] - 1, 1)  # Don't sample "".
      txt = labels[tf.random.uniform([], 0, num_texts, dtype=tf.int32)]
    else:
      txt = labels[0]  # Always works, since we append "" earlier on.

    token_ids = tokenizer.tokenize(txt[None])
    padded_token_ids, mask = tensorflow_text.pad_model_inputs(
        token_ids, max_len - 1)
    del mask  # Recovered from zero padding in model.
    count = tf.shape(padded_token_ids)[0]
    padded_token_ids = tf.concat(
        [tf.fill([count, 1], cls_token), padded_token_ids], axis=1)
    return padded_token_ids[0]

  return _pp_bert_tokenize
  