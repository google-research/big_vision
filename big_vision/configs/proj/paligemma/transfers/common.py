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

"""Common things across all transfer configs."""


TOKENIZER = 'gemma(tokensets=("loc", "seg"))'


def tok(**kw):
  """Creates the tokenization preprocessing string."""
  # Single entry point so that it's consistent everywhere and easier to switch.
  kw.setdefault('model', TOKENIZER)
  kw = ', '.join(f'{k}={repr(v)}' for k, v in kw.items())
  return f'tok({kw})'


def combine_and_keep_train(text_len, before=(), sep='\n'):
  return '|'.join([
      *before,
      tok(key='prefix', bos='yes'),
      tok(key='suffix', eos='yes'),
      tok(key='septok', text=sep),
      # If masks confuse you, see (internal link)
      'masked_concat(["prefix", "septok", "suffix"], mask_ar=[0, 0, 1], mask_loss=[0, 0, 1])',  # pylint: disable=line-too-long
      # For training, we +1 since the trainer removes EOS.
      f'tolen({text_len+1}, pad_value=0, key="text")',  # Value doesn't matter.
      f'tolen({text_len+1}, pad_value=1, key="mask_ar")',
      f'tolen({text_len+1}, pad_value=0, key="mask_loss")',
      'keep("image", "text", "mask_ar", "mask_loss")',
  ])


def combine_and_keep_eval(text_len, keep=tuple(), before=(), sep='\n'):
  return '|'.join([
      *before,
      # Same as training, except that suffix is now the empty string.
      # Meaning, we create text as [prefix separator pad],
      # and the mask accordingly as [0 0 1] (with repeats of respective lengths)
      tok(key='prefix', bos='yes'),
      tok(key='septok', text=sep),
      # If masks confuse you, see (internal link)
      'masked_concat(["prefix", "septok"], mask_ar=[0, 0], mask_input=[1, 1])',
      f'tolen({text_len}, pad_value=0, key="text")',  # value doesn't matter.
      f'tolen({text_len}, pad_value=1, key="mask_ar")',
      f'tolen({text_len}, pad_value=0, key="mask_input")',
      # And we need to keep everything that makes our evaluator happy.
      'keep(' + ', '.join(f'"{x}"' for x in (
          'image', 'text', 'mask_ar', 'mask_input') + tuple(keep)) + ')',
  ])
