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

"""Text-centric preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of (TF) tensors. The functors output a modified
dictionary.

A commonly used key for the tokenized output is "labels".
"""
import functools
import importlib

from absl import logging
from big_vision.datasets.imagenet import class_names as imagenet_class_names
from big_vision.pp import ops_general
from big_vision.pp import tokenizer as bv_tok
from big_vision.pp import utils
from big_vision.pp.registry import Registry
import tensorflow as tf

from tensorflow.io import gfile

import sentencepiece
SPProcessor = sentencepiece.SentencePieceProcessor

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sentencepiece.sentencepiece_model_pb2
del os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']
SPModelProto = sentencepiece.sentencepiece_model_pb2.ModelProto


# TODO: b/lbeyer - softly introduce and move to new tokenizer API.

KNOWN_TOKENIZERS = {
    "mc4":  # used in multilingual models (mT5, PaLI), vocab_size=250_000
        "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
    "cc_all":   # vocab_size=32_000
        "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model",
    "c4_en":   # vocab_size=32_000
        "gs://t5-data/vocabs/cc_en.32000/sentencepiece.model",
    "t5":  # same as cc_all, but with 100 extra dummy tokens used by T5 models
        "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model",
    "mt5":  # same as mc4, but with 100 extra dummy tokens used by T5 models
        "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
}


def create_tokenizer(model="c4_en", add_eos=True, add_bos=False):
  """Creates a tokenizer which can be used in tfds."""
  logging.info("Creating tokenizer: %s", model)
  with gfile.GFile(KNOWN_TOKENIZERS.get(model, model), "rb") as f:
    model = f.read()

  # Lazy import of tensorflow_text so it is an optional dependency for
  # the users of this file.
  import tensorflow_text
  return tensorflow_text.SentencepieceTokenizer(
      model=model, add_eos=add_eos, add_bos=add_bos
  )


def tokenize(input_text, tokenizer, max_len, *, pad_value, force_eos,
             multi_text=False):
  """Tokenizes string, and adds `pad_value` if longer than `max_len`."""

  def pad(tokens):
    # Truncate/pad to max_len.
    if force_eos:
      tokens = tf.cond(
          tf.shape(tokens)[0] >= max_len,
          lambda: tf.concat(
              # For too long, cut them off, but do keep the final EOS token.
              [tokens[:max_len - 1], tokens[-1:]], axis=0),
          lambda: tf.pad(
              tokens, [(0, max_len - tf.shape(tokens)[0])],
              constant_values=pad_value),
      )
    else:
      tokens = tokens[:max_len]
      tokens = tf.pad(
          tokens, [(0, max_len - tf.shape(tokens)[0])],
          constant_values=pad_value)
    tokens.set_shape([max_len])
    return tokens

  tokens = tokenizer.tokenize(input_text)

  if multi_text:
    tokens = tokens.to_tensor(pad_value)  # tf.RaggedTensor to tf.Tensor
    tokens = tf.reshape(tokens, [-1, tf.shape(tokens)[-1]])
    tokens = tf.map_fn(pad, tokens)  # `map_fn` only maps on axis 0

    final_shape = tf.concat([tf.shape(input_text), [max_len]], axis=0)
    return tf.reshape(tokens, final_shape)
  else:
    return pad(tokens)


@Registry.register("preprocess_ops.tokenize")
@utils.InKeyOutKey(indefault=None, outdefault="labels")
def get_pp_tokenize(
    max_len,
    eos,
    model="c4_en",
    lower=True,
    sample_if_multi=True,
    pad_value="<pad>",
    add_bos=False
):
  """Tokenizes a text.

  Let's assume max_len=3 and id("</s>")=1, id("a")=2, then we have

  1. `eos="none", pad_value=0`:
     - "a" -> [2, 0, 0]
     - "aa" -> [2, 2, 0]
     - "aaa" -> [2, 2, 2]

  2. `eos="yes", pad_value=0`:
     - "a" -> [2, 1, 0]
     - "aa" -> [2, 2, 1]
     - "aaa" -> [2, 2, 2]

     This is usually used with generative models that need to learn when to
     properly predict a "</s>" (when the sentence is finished) and when to
     abstain (when the sentence is truncated).

  3. `eos="sticky", pad_value=0`:
     - "a" -> [2, 1, 0]
     - "aa" -> [2, 2, 1]
     - "aaa" -> [2, 2, 1]

  4. `eos="sticky", pad_value=1`:
     - "a" -> [2, 1, 1]
     - "aa" -> [2, 2, 1]
     - "aaa" -> [2, 2, 1]

     This is traditionally used with contrastive models that use the last token
     for embeddings, similarly to "cls" tokens in BERT-style models.

  Args:
    max_len: maximum length of the tokenized text.
    eos: Whether to add an "</s>" (end of sentence) token and whether to keep it
      when the sequence is longer than `max_len - 1`. See examples above for
      details. Valid values: "none", "yes", "sticky".
    model: a path to the pretrained sentencepiece model.
    lower: lowercase the text before tokenizing.
    sample_if_multi: If there's more than one, randomly pick one if this is
      True; otherwise pick all texts and keep the input's batch shape in result.
    pad_value: which token to pad the sequence with. If a string (for example
      `"<pad>"`), tokenize it and use its first token. Note that there is no
      guarantee to have any padding at the end of the sentence, if the sentence
      is longer than `max_len`.
    add_bos: adds beginning of sentence symbol.

  Returns:
    an op that outputs tokenized text.
  """

  if eos not in ("yes", "none", "sticky"):
    raise ValueError(f"Invalid value for eos: '{eos}'.")

  tokenizer = create_tokenizer(model, add_eos=eos != "none", add_bos=add_bos)

  if isinstance(pad_value, str):
    pad_value = tokenizer.string_to_id(pad_value)

  def _pp_tokenize(txt):
    if sample_if_multi and tf.convert_to_tensor(txt).ndim:
      # TODO: I wish this code-path could die.
      logging.warning("sample_if_multi is deprecated and will be removed."
                      "Call `choice` (and maybe `setdefault`) instead.")
      txt = ops_general.get_choice(key="t")(
          ops_general.get_setdefault("t", "")({"t": txt}))["t"]

    if lower:
      txt = tf.strings.lower(txt) if sample_if_multi else tf.map_fn(
          tf.strings.lower, txt)

    return tokenize(
        txt,
        tokenizer,
        max_len,
        pad_value=pad_value,
        force_eos=eos == "sticky",
        multi_text=not sample_if_multi)

  return _pp_tokenize


@Registry.register("preprocess_ops.coco_captions")
def get_coco_captions(outkey="captions"):
  """Extracts coco's captions from nested dict."""

  def _pp_coco_captions(data):
    data[outkey] = data["captions"]["text"]
    return data

  return _pp_coco_captions


@Registry.register("preprocess_ops.clip_i1k_label_names")
@utils.InKeyOutKey(indefault="label", outdefault="labels")
def get_pp_clip_i1k_label_names():
  """Convert i1k label numbers to strings, using CLIP's class names."""

  def _pp_imagenet_labels(label):
    return tf.gather(imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES, label)

  return _pp_imagenet_labels


@Registry.register("preprocess_ops.lower")
@utils.InKeyOutKey(indefault="text", outdefault="text")
def get_lower():
  """Lowercases text feature."""

  def _pp_lower(text):
    return tf.strings.lower(text)

  return _pp_lower


def _add_pieces(model_bytes, extra_pieces):
  """Adds extra pieces to sentencpiece model specified by `model_bytes`."""

  model = SPProcessor()
  model.LoadFromSerializedProto(model_bytes)
  unk_idx = model.PieceToId("<unk>")
  assert model.IdToPiece(unk_idx) == "<unk>", model.IdToPiece(unk_idx)

  model_proto = SPModelProto.FromString(model_bytes)
  idx_to_updated_piece = {}
  for piece in extra_pieces:
    # The SentencePieceModel proto stores whitespaces as the special
    # character '▁'. We perform the conversion here.
    piece = piece.replace(" ", "▁")
    spiece = model_proto.SentencePiece(
        piece=piece,
        # We set the highest score to force priority on user defined tokens.
        score=0.0,
        type=model_proto.SentencePiece().Type.USER_DEFINED,
    )
    existing_idx = model.PieceToId(piece)
    if (existing_idx != unk_idx) ^ (piece == "<unk>"):
      idx_to_updated_piece[existing_idx] = spiece
      logging.info("Updating token at idx %d: %s", existing_idx, spiece.piece)
    else:
      model_proto.pieces.append(spiece)

  # Replace duplicated pieces with updated ones.
  updated_pieces = [
      idx_to_updated_piece.get(i, piece)
      for i, piece in enumerate(model_proto.pieces)
  ]
  del model_proto.pieces[:]
  model_proto.pieces.extend(updated_pieces)

  return model_proto.SerializeToString()


def _iterable(x):
  if isinstance(x, tf.RaggedTensor):
    return True
  if getattr(x, "ndim", 0) > 1:  # np, jnp
    return True
  if isinstance(x, (list, tuple)) and not isinstance(x[0], (int, float)):
    return True
  return False


@Registry.register("tokenizers.sp")
class SentencepieceTokenizer(bv_tok.Tokenizer):
  """Wraps a `tftext.SentencepieceTokenizer`.

  If you plan to use this tokenizer, please familiarize yourself with the test
  cases first. This is likely to save you a lot of troubles down the road, trust
  me!
  """

  def __init__(self, model, tokensets=()):
    with gfile.GFile(KNOWN_TOKENIZERS.get(model, model), "rb") as f:
      model_bytes = f.read()
    extras = bv_tok.get_extra_tokens(tokensets)
    model_bytes = _add_pieces(model_bytes, extras)
    self._tok_sp = SPProcessor()
    self._tok_sp.LoadFromSerializedProto(model_bytes)
    self.extras = {self._tok_sp.PieceToId(x): x for x in extras}

  def to_int(self, text, *, bos=False, eos=False):
    def _single(s):
      return (
          ([self.bos_token] if bos else []) +
          self._tok_sp.EncodeAsIds(s) +
          ([self.eos_token] if eos else [])
      )
    if isinstance(text, str):
      return _single(text)
    return type(text)([_single(s) for s in text])

  def to_str(self, tokens, *, stop_at_eos=True):
    def _single(toks):
      toks = [int(t) for t in toks]  # We really need this for DecodeIds.
      if stop_at_eos:
        try:  # The SentencePiece strips eos, but does not stop at it, so we do.
          toks = toks[:toks.index(self.eos_token)]
        except ValueError:  # No eos token found, nothing to do.
          pass
      return self._tok_sp.DecodeIds(toks)
    if _iterable(tokens):
      return [_single(toks) for toks in tokens]
    return _single(tokens)

  def _check_known(self, piece):
    if (id_ := self._tok_sp.PieceToId(piece)) == self._tok_sp.unk_id():
      logging.error("Piece '%s' is not known (unk=%s)!", piece, id_)
    return id_

  def to_piece(self, idx):
    return self._tok_sp.IdToPiece(int(idx))

  @property
  def pad_token(self):
    return self._tok_sp.pad_id()

  @property
  def eos_token(self):
    return self._tok_sp.eos_id()

  @property
  def bos_token(self):
    return self._tok_sp.bos_id()

  @property
  def vocab_size(self):
    return self._tok_sp.GetPieceSize()

  # For the _tf_op variants, we need a lot of wrapping boilerplate.

  def to_int_tf_op(self, text, *, bos=False, eos=False):
    text = tf.convert_to_tensor(text)
    if text.ndim == 0:
      def fn(txt):
        string = txt.numpy().decode()
        return tf.constant(self.to_int(string, bos=bos, eos=eos), tf.int32)
      return tf.py_function(fn, [text], tf.int32)
    else:
      def fn(txt):
        strings = [s.decode() for s in txt.numpy().tolist()]
        toks = self.to_int(strings, bos=bos, eos=eos)
        return tf.ragged.constant(toks)
      out_type = tf.RaggedTensorSpec([tf.shape(text)[0], None], tf.int32)
      return tf.py_function(fn, [text], Tout=out_type)

  def to_str_tf_op(self, tokens, *, stop_at_eos=True):
    def single(t):
      fn = functools.partial(self.to_str, stop_at_eos=stop_at_eos)
      return tf.numpy_function(fn, [t], tf.string, stateful=False)
    if _iterable(tokens):
      return tf.map_fn(single, tokens, tf.string)
    return single(tokens)
