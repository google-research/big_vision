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

"""Tests for ops_text."""

import copy

from absl.testing import parameterized
import big_vision.pp.ops_text as pp
from big_vision.pp.registry import Registry
import numpy as np
import tensorflow as tf


class PyToTfWrapper:
  """Allows to use `to_{int,str}_tf()` via `to_{int,str}()`."""

  def __init__(self, tok):
    self.tok = tok
    self.bos_token = tok.bos_token
    self.eos_token = tok.eos_token
    self.vocab_size = tok.vocab_size

  def to_int(self, text, *, bos=False, eos=False):
    ret = self.tok.to_int_tf_op(text, bos=bos, eos=eos)
    if isinstance(ret, tf.RaggedTensor):
      return [t.numpy().tolist() for t in ret]
    return ret.numpy().tolist()

  def to_str(self, tokens, stop_at_eos=True):
    ret = self.tok.to_str_tf_op(
        tf.ragged.constant(tokens),
        stop_at_eos=stop_at_eos,
    )
    if ret.ndim == 0:
      return ret.numpy().decode()
    return [t.numpy().decode() for t in ret]


class PpOpsTest(tf.test.TestCase, parameterized.TestCase):

  def tfrun(self, ppfn, data):
    # Run once as standalone, as could happen eg in colab.
    yield {k: np.array(v) for k, v in ppfn(copy.deepcopy(data)).items()}

    # And then once again as part of tfdata pipeline.
    # You'd be surprised how much these two differ!
    tfdata = tf.data.Dataset.from_tensors(copy.deepcopy(data))
    for npdata in tfdata.map(ppfn).as_numpy_iterator():
      yield npdata

  def testtok(self):
    # https://github.com/google/sentencepiece/blob/master/python/test/test_model.model
    return "test_model.model"  # Should we just commit it? It's 200kB

  def test_get_pp_clip_i1k_label_names(self):
    op = pp.get_pp_clip_i1k_label_names()
    labels = op({"label": tf.constant([0, 1])})["labels"].numpy().tolist()
    self.assertAllEqual(labels, ["tench", "goldfish"])

  @parameterized.parameters((b"Hello world ScAlAr!", b"hello world scalar!"),
                            (["Decoded Array!"], ["decoded array!"]),
                            ([b"aA", "bB"], [b"aa", "bb"]))
  def test_get_lower(self, inputs, expected_output):
    op = pp.get_lower()
    out = op({"text": tf.constant(inputs)})
    self.assertAllEqual(out["text"].numpy(), np.array(expected_output))

  @parameterized.named_parameters(
      ("py", False),
      ("tf", True),
  )
  def test_sentencepiece_tokenizer(self, wrap_tok):
    tok = pp.SentencepieceTokenizer(self.testtok())
    if wrap_tok:
      tok = PyToTfWrapper(tok)
    self.assertEqual(tok.vocab_size, 1000)
    bos, eos = tok.bos_token, tok.eos_token
    self.assertEqual(bos, 1)
    self.assertEqual(eos, 2)
    # Note: test model does NOT have a <pad> token (similar to e.g. "mistral").
    # `.to_int()` wraps `.to_int_tf_ops` which is thus also tested
    self.assertEqual(tok.to_int("blah"), [80, 180, 60])
    self.assertEqual(tok.to_int("blah", bos=True), [bos, 80, 180, 60])
    self.assertEqual(tok.to_int("blah", eos=True), [80, 180, 60, eos])
    self.assertEqual(
        tok.to_int("blah", bos=True, eos=True), [bos, 80, 180, 60, eos]
    )
    self.assertEqual(
        tok.to_int(["blah", "blah blah"]),
        [[80, 180, 60], [80, 180, 60, 80, 180, 60]],
    )
    # inverse of above
    # `.to_str()` wraps `.to_str_tf_ops` which is thus also tested
    self.assertEqual(tok.to_str([80, 180, 60]), "blah")
    self.assertEqual(tok.to_str([1, 80, 180, 60]), "blah")
    self.assertEqual(tok.to_str([80, 180, 60, 2]), "blah")
    self.assertEqual(
        tok.to_str([[80, 180, 60], [80, 180, 60, 80, 180, 60]]),
        ["blah", "blah blah"],
    )

  def test_sentencepiece_tokenizer_tf_op_ndarray_input(self):
    tok = pp.SentencepieceTokenizer(self.testtok())
    bos, eos = tok.bos_token, tok.eos_token
    arr = np.array([[bos, 80, 180, 60, eos]] * 2, dtype=np.int32)
    self.assertEqual(tok.to_str_tf_op(arr).numpy().tolist(), [b"blah"] * 2)

  def test_sentencepiece_tokenizer_tokensets(self):
    tok = pp.SentencepieceTokenizer(self.testtok(), tokensets=["loc"])
    self.assertEqual(tok.vocab_size, 2024)
    self.assertEqual(
        tok.to_int("blah<loc0000><loc1023>"), [80, 180, 60, 1000, 2023]
    )

  def test_sentencepiece_stop_at_eos(self):
    tok = pp.SentencepieceTokenizer(self.testtok())
    self.assertEqual(tok.to_str([80, 180, 60], stop_at_eos=False), "blah")
    eos = tok.eos_token
    self.assertEqual(tok.to_str([80, eos, 180, 60], stop_at_eos=False), "blah")
    self.assertEqual(tok.to_str([80, eos, 180, 60], stop_at_eos=True), "b")
    self.assertEqual(
        tok.to_str([[80, eos, 180, 60], [80, 180, eos, 60]], stop_at_eos=True),
        ["b", "bla"]
    )

  def test_sentencepiece_extra_tokens(self):
    tok = pp.SentencepieceTokenizer(self.testtok())
    self.assertEqual(tok.to_str([1, 80, 180, 60, 2], stop_at_eos=False), "blah")
    tok = pp.SentencepieceTokenizer(
        self.testtok(), tokensets=["sp_extra_tokens"]
    )
    self.assertEqual(tok.vocab_size, 1001)  # Also added the <pad> token.
    self.assertEqual(
        tok.to_str([1, 80, 180, 60, 2], stop_at_eos=False), "<s> blah</s>"
    )


@Registry.register("tokensets.sp_extra_tokens")
def _get_sp_extra_tokens():
  # For sentencepiece, adding these tokens will make them visible when decoding.
  # If a token is not found (e.g. "<pad>" is not found in "mistral"), then it is
  # added to the vocabulary, increasing the vocab_size accordingly.
  return ["<s>", "</s>", "<pad>"]


if __name__ == "__main__":
  tf.test.main()
