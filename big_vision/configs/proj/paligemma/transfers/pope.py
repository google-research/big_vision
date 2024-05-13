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

# pylint: disable=line-too-long
r"""PaliGemma (0-shot) evaluation in POPE.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

_DATASETS = ('pope_random', 'pope_popular', 'pope_adversarial')


# Note that POPE does not have training data, only test data.
# We're defining training_data() here anyway for symmetry with the other
# transfers. We will train for 0 steps on this data, i.e. not at all.
def training_data(res, *, text_len, prefix):
  """Creates training data config.

  See (internal link)

  Args:
    res: The requested image resolution (eg 224).
    text_len: sequence length.
    prefix: prefix to use in the prompt: (e.g. 'answer en {question}')

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  pp = '|'.join([
      f'decode|resize({res})|value_range(-1, 1)',
      f'strfmt("{prefix}", outkey="prefix")',
      'copy(inkey="answer", outkey="suffix")',
      combine_and_keep_train(text_len=text_len),
  ])
  c.data = {}
  for dataset in _DATASETS:
    c.data[dataset] = 1  # Set weight to 1.
    c[dataset] = dict(
        pp=pp,
        data=dict(
            name=dataset,
            split='test',
        ),
    )
  return c


def add_eval(c, res, *, text_len, prefix):
  """Add eval configs."""
  pp_eval = '|'.join([
      f'decode|resize({res})|value_range(-1, 1)',
      f'strfmt("{prefix}", outkey="prefix")',
      combine_and_keep_eval(text_len=text_len, keep=('question_id', 'answer')),
  ])
  for dataset in _DATASETS:
    c.evals[f'pope/{dataset}'] = dict(
        type='proj.paligemma.transfers.pope',
        pred='decode', pred_kw={'max_decode_len': text_len},
        log_percent=1, tokenizer=TOKENIZER,
        data=dict(
            name=dataset,
            split='test',
        ),
        pp_fn=pp_eval,
    )


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', res=224, text_len=48, prefix='{question}')

  c.name = ''  # Help to track experiments.
  c.input = training_data(c.res, text_len=c.text_len, prefix=c.prefix)

  # Make the config eval-only by setting some dummies.
  c.total_steps = 0
  c.input.batch_size = 256
  c.optax_name = 'identity'
  c.lr = 0.0

  # Add evaluators.
  c.evals = {}
  add_eval(c, c.res, text_len=c.text_len, prefix=c.prefix)

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0)
  c.model_init = f'pt_{c.res}'

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  # These probably do not need any change/tuning
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  # Update configs for quicker local runs and avoid swapping.
  if c.mode in ('runlocal', 'mock'):
    for ev in c.evals.values():
      ev.data.split = ev.data.split.split('[')[0] + '[:16]'

  return c