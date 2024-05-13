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

"""The tokenizer API for big_vision, and central registration place."""
import functools
import importlib
from typing import Protocol

from absl import logging
from big_vision.pp import registry
import big_vision.utils as u
import numpy as np


class Tokenizer(Protocol):
  """Just to unify on the API as we now have mmany different ones."""

  def to_int(self, text, *, bos=False, eos=False):
    """Tokenizes `text` into a list of integer tokens.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      List or list-of-list of tokens.
    """

  def to_int_tf_op(self, text, *, bos=False, eos=False):
    """Same as `to_int()`, but as TF ops to be used in pp."""

  def to_str(self, tokens, *, stop_at_eos=True):
    """Inverse of `to_int()`.

    Args:
      tokens: list of tokens, or list of lists of tokens.
      stop_at_eos: remove everything that may come after the first EOS.

    Returns:
      A string (if `tokens` is a list of tokens), or a list of strings.
      Note that most tokenizers strip select few control tokens like
      eos/bos/pad/unk from the output string.
    """

  def to_str_tf_op(self, tokens, *, stop_at_eos=True):
    """Same as `to_str()`, but as TF ops to be used in pp."""

  @property
  def pad_token(self):
    """Token id of padding token."""

  @property
  def eos_token(self):
    """Token id of end-of-sentence token."""

  @property
  def bos_token(self):
    """Token id of beginning-of-sentence token."""

  @property
  def vocab_size(self):
    """Returns the size of the vocabulary."""


@functools.cache
def get_tokenizer(name):
  with u.chrono.log_timing(f"z/secs/tokenizer/{name}"):
    if not registry.Registry.knows(f"tokenizers.{name}"):
      raw_name, *_ = registry.parse_name(name)
      logging.info("Tokenizer %s not registered, "
                   "trying import big_vision.pp.%s", name, raw_name)
      importlib.import_module(f"big_vision.pp.{raw_name}")

    return registry.Registry.lookup(f"tokenizers.{name}")()


def get_extra_tokens(tokensets):
  extra_tokens = []
  for tokenset in tokensets:
    extra_tokens.extend(registry.Registry.lookup(f"tokensets.{tokenset}")())
  return list(np.unique(extra_tokens))  # Preserves order. Dups make no sense.


@registry.Registry.register("tokensets.loc")
def _get_loc1024(n=1024):
  return [f"<loc{i:04d}>" for i in range(n)]


@registry.Registry.register("tokensets.seg")
def _get_seg(n=128):
  return [f"<seg{i:03d}>" for i in range(n)]
