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

"""Utilities for fake BERT checkpoint."""

import tempfile

import tensorflow.compat.v1 as tf

# Checkpoint structure was extracted with the following (Colab) snippet:
#
# !wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip  # pylint: disable=line-too-long
# !unzip uncased_L-12_H-768_A-12.zip
#
# import tensorflow.compat.v1 as tf
#
# ckpt_reader = tf.train.load_checkpoint('bert_model.ckpt')
# tf_params = {
#     tf_name: ckpt_reader.get_tensor(tf_name)
#     for tf_name in ckpt_reader.get_variable_to_dtype_map()
# }
#
# 'shapes_dtypes = {\n%s\n}' % '\n'.join(
#     f'  "{k}": ({v.shape}, "{v.dtype}"),'
#     for k, v, in tf_params.items()
# )

# pylint: disable=line-too-long
_BASE_SHAPES_DTYPES = {
    "cls/seq_relationship/output_bias": ((2,), "float32"),
    "cls/predictions/transform/LayerNorm/gamma": ((768,), "float32"),
    "cls/predictions/transform/LayerNorm/beta": ((768,), "float32"),
    "bert/pooler/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_9/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_9/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_3/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_7/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_9/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_7/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_9/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_9/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_9/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_9/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_9/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_8/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_4/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_8/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_8/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_11/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_11/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_8/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_8/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_2/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_8/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_1/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_8/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_3/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_8/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_8/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_8/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_7/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_7/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_8/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_7/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_7/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_9/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_7/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_7/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_6/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_6/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_6/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_0/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_6/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_7/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_4/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_2/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_5/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_5/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_9/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_3/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_8/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_5/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_5/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_5/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_4/output/dense/bias": ((768,), "float32"),
    "bert/embeddings/token_type_embeddings": ((2, 768), "float32"),
    "bert/encoder/layer_4/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_4/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_7/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_4/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_9/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_10/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_6/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_4/attention/self/query/bias": ((768,), "float32"),
    "cls/seq_relationship/output_weights": ((2, 768), "float32"),
    "bert/encoder/layer_7/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_4/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_4/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_4/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_3/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_1/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_2/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_8/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_4/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_3/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_4/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_3/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_1/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_3/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_10/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_3/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_1/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_0/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_10/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_3/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_3/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_1/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_3/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_1/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_3/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_2/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_6/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_11/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_2/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_2/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_2/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_2/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_6/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_11/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_6/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_11/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_11/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_11/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_10/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_11/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_6/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_6/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_11/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_10/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_4/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_11/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_10/attention/self/query/bias": ((768,), "float32"),
    "bert/embeddings/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_2/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_11/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_11/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_5/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_3/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_10/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_10/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/embeddings/word_embeddings": ((30522, 768), "float32"),
    "bert/encoder/layer_9/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_9/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_6/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_10/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_6/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_1/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_5/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_2/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_0/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_3/intermediate/dense/kernel": ((768, 3072), "float32"),
    "cls/predictions/output_bias": ((30522,), "float32"),
    "bert/encoder/layer_0/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_6/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_0/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_2/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_10/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_5/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_4/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_0/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_0/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_10/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_7/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_3/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_2/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_8/output/dense/kernel": ((3072, 768), "float32"),
    "bert/embeddings/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_1/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_10/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_2/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_6/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_2/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_11/attention/self/value/bias": ((768,), "float32"),
    "bert/encoder/layer_9/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_0/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_10/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_10/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_1/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_8/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_0/intermediate/dense/bias": ((3072,), "float32"),
    "bert/encoder/layer_1/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_1/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_7/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_2/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_8/attention/output/dense/bias": ((768,), "float32"),
    "cls/predictions/transform/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_6/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_5/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_0/attention/self/value/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_7/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_7/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_1/output/dense/kernel": ((3072, 768), "float32"),
    "bert/encoder/layer_11/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_4/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_1/attention/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_9/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_2/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_0/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_10/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_1/attention/self/query/bias": ((768,), "float32"),
    "bert/encoder/layer_3/output/LayerNorm/beta": ((768,), "float32"),
    "bert/encoder/layer_6/attention/output/dense/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_1/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_11/output/dense/bias": ((768,), "float32"),
    "cls/predictions/transform/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_0/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_11/attention/self/query/kernel": ((768, 768), "float32"),
    "bert/encoder/layer_0/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_0/attention/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_7/attention/output/LayerNorm/gamma": ((768,), "float32"),
    "bert/encoder/layer_4/attention/self/key/bias": ((768,), "float32"),
    "bert/encoder/layer_10/attention/self/key/kernel": ((768, 768), "float32"),
    "bert/embeddings/position_embeddings": ((512, 768), "float32"),
    "bert/encoder/layer_1/output/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_9/intermediate/dense/kernel": ((768, 3072), "float32"),
    "bert/encoder/layer_0/output/LayerNorm/beta": ((768,), "float32"),
    "bert/pooler/dense/bias": ((768,), "float32"),
    "bert/encoder/layer_0/attention/output/LayerNorm/beta": ((768,), "float32"),
}
# pylint: enable=line-too-long


def create_base_checkpoint():
  """Returns path to fake Bert "base" checkpoint directory (zero init)."""
  directory = tempfile.mkdtemp()
  path = f"{directory}/bert_model.ckpt"
  with tf.Session() as sess:
    for name, (shape, dtype) in _BASE_SHAPES_DTYPES.items():
      tf.Variable(tf.zeros(shape, dtype), name=name)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, path)
  return directory
