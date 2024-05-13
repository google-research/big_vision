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

"""pp ops."""

from big_vision.pp.registry import Registry
import tensorflow as tf


@Registry.register('preprocess_ops.sci_qa_choices_shuffle')
def sci_qa_choices_shuffle(
    choice_str_inkey='choices',
    ans_inkey='answer',
    indexed_choices_outkey='indexed_choices',
    indexed_answer_outkey='indexed_answer',
):
  """Random shuffle the sci_qa's choice on the fly.

  Args:
    choice_str_inkey: the original choice list from
      sciqa,e.g['apple','banana',..]
    ans_inkey: the original answer from sciqa e.g. 1
    indexed_choices_outkey: shuffled choice (with index suffix concat to string)
      e.g."(A) banana, (B) apple"
    indexed_answer_outkey: shuffled answer with abc index, e,g
      1(original)->2(shuffled)->'B' (alphabet index)

  Returns:
  """
  def _template(data):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    abc_tensor = tf.constant([f'({a})' for a in alphabet])
    abcans_tensor = tf.constant([f'{a}' for a in alphabet])
    choices = data[choice_str_inkey]
    indices = tf.range(len(choices))
    # Shuffle the indices
    shuffled_indices = tf.random.shuffle(indices)
    # Use the shuffled indices to shuffle the tensor
    shuffled_tensor = tf.gather(choices, shuffled_indices)

    abc_tensor = tf.gather(abc_tensor, indices)

    data[indexed_choices_outkey] = tf.strings.reduce_join(
        tf.strings.join([abc_tensor, shuffled_tensor], separator=' '),
        separator=', ',
    )

    answer_tensor = data[ans_inkey]
    new_ans_indice = tf.where(tf.equal(shuffled_indices, answer_tensor))
    new_ans_indice = tf.gather(abcans_tensor, new_ans_indice)
    data[indexed_answer_outkey] = tf.strings.reduce_join(new_ans_indice)
    return data

  return _template
