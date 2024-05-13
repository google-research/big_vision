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

import functools
import string

from big_vision.pp import ops_text
from big_vision.pp import utils
from big_vision.pp.registry import Registry
import big_vision.pp.tokenizer as bv_tok
import numpy as np
import tensorflow as tf


@Registry.register('tokenizers.gemma')
def get_tokenizer_gemma(
    tokensets=(),
    model='gs://big_vision/gemma_tokenizer.model',
):
  # See (internal link) for colab playground.
  return ops_text.SentencepieceTokenizer(model=model, tokensets=tokensets)


@functools.cache
def tokenize_constant(model, text, bos='no', eos='no', length=None):
  """Tokenize a constant string, with memoization."""
  assert eos in ('no', 'yes', 'sticky')
  assert bos in ('no', 'yes')
  tokenizer = bv_tok.get_tokenizer(model)
  tokens = tokenizer.to_int(
      text, bos=bos == 'yes', eos=eos in ('yes', 'sticky'))

  if length is None:
    return tokens

  if len(tokens) > length:
    if eos == 'sticky':
      return np.r_[tokens[:length-1], tokens[-1]]
    else:
      return tokens[:length]
  else:
    return np.pad(tokens, [(0, length - len(tokens))],
                  constant_values=tokenizer.pad_token)


@Registry.register('preprocess_ops.tolen')
@utils.InKeyOutKey(indefault=None, outdefault=None, with_data=True)
def get_tolen(length, *, sticky_end=False, pad_value=None, pad_key=None):
  """Gets token to a fixed length."""
  def _tolen(x, data):
    if not length:
      return x

    xlen = tf.shape(x)[0]

    if sticky_end:
      trunc_fn = lambda: tf.concat([x[:length - 1], x[-1:]], axis=0)
    else:
      trunc_fn = lambda: x[:length]

    # Potentially get the pad value from a data key (to be tokenizer agnostic).
    pad_value_ = pad_value
    if pad_key:
      pad_value_ = data[pad_key]
      # If coming from a previous tokenization op, it's probably 1D; take first.
      if getattr(pad_value_, 'ndim', 0) == 1:
        pad_value_ = pad_value_[0]
    assert pad_value_ is not None, 'Need either pad_value or pad_key.'

    pad_fn = lambda: tf.pad(x, [(0, length - xlen)], constant_values=pad_value_)
    out = tf.cond(xlen >= length, trunc_fn, pad_fn)
    out.set_shape([length])
    return out
  return _tolen


@Registry.register('preprocess_ops.tok')
def get_tokenize(model, length=None, *, bos='no', eos='no',
                 text=None, key=None, inkey=None, outkey=None):
  """Tokenizes and optionally truncates/pads a string."""

  assert eos in ('no', 'yes', 'sticky')
  assert bos in ('no', 'yes')
  outkey_ = outkey or key
  inkey_ = inkey or key

  if text is not None:
    assert inkey is None, 'Either inkey or text, not both.'
    tokens = tokenize_constant(model, text, bos=bos, eos=eos, length=length)
    def _pp_tokenize_text(data):
      data[outkey_] = tokens
      return data
    return _pp_tokenize_text

  tokenizer = bv_tok.get_tokenizer(model)

  def _pp_tokenize(data):
    assert getattr(data[inkey_], 'ndim', 0) == 0, (
        f'Can only tokenize single string ({inkey_}, {data[inkey_].ndim}-D)')

    toks = tokenizer.to_int_tf_op(
        data[inkey_], bos=bos == 'yes', eos=eos in ('yes', 'sticky'))

    tolen = get_tolen(
        length, sticky_end=eos == 'sticky',
        pad_value=bv_tok.get_tokenizer(model).pad_token,
        key='tmp',
    )
    toks = tolen({'tmp': toks})['tmp']

    data[outkey_] = toks
    return data
  return _pp_tokenize


@Registry.register('preprocess_ops.masked_concat')
def get_masked_concat(keys, outkey='text', **masks):
  assert all(len(keys) == len(m) for m in masks.values()), (keys, masks)
  def _masked_concat(data):
    data[outkey] = tf.concat([data[k] for k in keys], axis=0)
    for mask_name, mask_vals in masks.items():
      m = [tf.fill(tf.shape(data[k]), v) for k, v in zip(keys, mask_vals)]
      data[mask_name] = tf.concat(m, axis=0)
    return data
  return _masked_concat


@Registry.register('preprocess_ops.strfmt')
def get_strfmt(template, outkey='text'):
  """Formats a string template with content form the data dict."""

  def _template(data):
    outputs = []
    parts = string.Formatter().parse(template)
    for (literal_text, field_name, format_spec, conversion) in parts:
      # For now, we keep it simple and don't support fancy format specs.
      # But we can add support to that via py_func as soon as we need it.
      assert not format_spec and not conversion
      outputs.append(tf.constant(literal_text))
      if field_name:
        value = data[field_name]
        # Convert any non-strings (numbers, vectors) to a string.
        if tf.convert_to_tensor(value).dtype != tf.string:
          value = tf.strings.format('{}', value, summarize=-1)
        outputs.append(value)
    data[outkey] = tf.strings.join(outputs)
    return data

  return _template


@Registry.register('preprocess_ops.strjoin')
@utils.InKeyOutKey()
def get_strjoin(glue):
  def _strjoin(x):
    return tf.strings.reduce_join(x, separator=glue)
  return _strjoin


@Registry.register('preprocess_ops.majority')
@utils.InKeyOutKey()
def get_majority():
  def _majority(x):
    val, _, count = tf.unique_with_counts(x)  # Sadly, stablesorted.
    return val[tf.argmax(count)]
  return _majority


@Registry.register('preprocess_ops.getidx')
def getidx(inkey, index_key, outkey=None):
  """Indexes a tensor and stores result in outkey."""
  def _getidx(data):
    idx = data[index_key]
    array = data[inkey]
    data[outkey or inkey] = array[idx]
    return data
  return _getidx
