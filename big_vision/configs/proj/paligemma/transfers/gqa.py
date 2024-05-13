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
r"""PaliGemma transfer to GQA (https://arxiv.org/abs/1902.09506).
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

XGQA_LANGUAGES = ('bn', 'de', 'en', 'id', 'ko', 'pt', 'ru', 'zh')


def training_data(res, *, final_split, prefix, text_len=32):
  """Creates training data config.

  See (internal link)
  You can add more arguments beside `res`, but give them good defaults.

  Args:
    res: The requested image resolution (eg 224).
    final_split: Whether to train on train+val.
    prefix: The prefix to use for the input. E.g. "answer en {question}"
    text_len: sequence length.

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='gqa',
      split='train_balanced+val_balanced' if final_split else 'train_balanced',
  )
  c.pp = '|'.join([
      f'decode|resize({res})|value_range(-1, 1)',
      f'strfmt("{prefix}", outkey="prefix")',
      'copy(inkey="answer", outkey="suffix")',
      combine_and_keep_train(text_len),
  ])
  return c


def add_eval(c, res, *, text_len=32, prefix, **kw):
  """GQA evaluators."""
  c_train = training_data(res, final_split=True, prefix=prefix, text_len=text_len)

  pp = '|'.join([
      f'decode|resize({res})|value_range(-1, 1)',
      'copy(inkey="example_id", outkey="question_id")',
      # GQA: both questions and answers are always in english.
      # xGQA: questions in different languages. Answers always in english.
      f'strfmt("{prefix}", outkey="prefix")',
      combine_and_keep_eval(text_len, keep=('answer', 'question_id')),
  ])

  for freq, name, split, skip_first in [
      # TODO: adjust the proportion of dataset seen in these minivals
      #   based speed on hardware.
      (1/8, 'minitrain', 'train_balanced[:10000]', False),  # To gauge memorization.
      (1/8, 'val_balanced', 'val_balanced', True),          # To tune hparams.
      (1.0, 'testdev_balanced', 'testdev_balanced', True),  # To compute final publishable scores.
  ]:
    c.evals[f'gqa/{name}/decode'] = dict(
        type='proj.paligemma.transfers.vqa',
        pred='decode', pred_kw={'max_decode_len': text_len},
        outfile=f'{{workdir}}/gqa_{name}.json',
        out_question_key='question_id', out_answer_key='prediction',
        data={**c_train.data, 'split': split},
        log_percent=freq, skip_first=skip_first, tokenizer=TOKENIZER, pp_fn=pp)
    c.evals[f'gqa/{name}/decode'].update(kw)

  # Add XGQA evaluators. Zero shot since the model is trained only in GQA (en).
  for lang in XGQA_LANGUAGES:
    c.evals[f'xgqa/test_zs_{lang}/decode'] = dict(
        type='proj.paligemma.transfers.vqa',
        pred='decode', pred_kw={'max_decode_len': text_len},
        outfile=f'{{workdir}}/xgqa_test_{lang}.json',
        data=dict(
            name='xgqa',
            split=f'test_zs_{lang}',  # Zero-shot split
        ),
        log_percent=1/8, tokenizer=TOKENIZER, pp_fn=pp)
    c.evals[f'xgqa/test_zs_{lang}/decode'].update(kw)


def add_eval_pplx(c, res, *, text_len=32, prefix):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, final_split=True, text_len=text_len, prefix=prefix)
  for name, split in [
      ('minitrain', 'train_balanced[:5%]'),  # To gauge memorization.
      ('minival', 'val_balanced[:5%]'),  # To tune hparams.
  ]:
    c.evals[f'gqa/{name}/pplx'] = dict(
        type='proj.paligemma.perplexity', pred='logits',
        key='text', shift_labels=True,
        log_percent=0.05,  # Eval ~20x per run; it's cheap.
        data={**c_train.data, 'split': split},
        pp_fn=c_train.pp,
    )


def sweep_best(add, arg=None):
  """Train with best hyper-params."""
  c = bvcc.parse_arg(arg, final_split=False)
  # Based on (internal link), (internal link), (internal link).
  # TODO: Is there a more compreensive sweep and can we use
  # freeze_vit=False for all resolutions (and more common in other configs)?
  add(lr=1e-5, wd=0.0, **bvcc.arg(res=224, freeze_vit=False, **c))
  add(lr=1e-5, wd=0.0, **bvcc.arg(res=448, freeze_vit=True, **c))
  # Not better: add(lr=1e-5, wd=0.0, **bvcc.arg(res=896, freeze_vit=True, **c))


sweep = sweep_best  # Choose which sweep to run.


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', res=224, final_split=False,
                     freeze_vit=True, freeze_llm=False,
                     prefix='answer en {question}')

  c.name = ''
  c.input = training_data(c.res, final_split=c.final_split, prefix=c.prefix)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 1
  c.input.batch_size = 256
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 1e-5
  c.wd = 0.0
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
  ]

  # Add evaluators.
  c.evals = {}
  add_eval(c, c.res, batch_size=1024, prefix=c.prefix)
  add_eval_pplx(c, c.res, prefix=c.prefix)

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
  c.input.shuffle_buffer_size = 50_000
  c.log_training_steps = 50
  c.ckpt_steps = 1_000
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  # Update configs for quicker local runs and avoid swapping.
  if c.mode in ('runlocal', 'mock'):
    c.input.shuffle_buffer_size = None
    for ev in c.evals.values():
      ev.data.split = ev.data.split.split('[')[0] + '[:16]'

  if c.mode == 'runlocal':
    c.log_training_steps = 1
    c.input.batch_size = 2

  c.seed = 0
  return c


def metrics():
  m = ['training_loss']
  m.append('gqa/minitrain/pplx/avg')
  m.append('gqa/minival/pplx/avg')
  m.append('gqa/minitrain/decode/acc')
  m.append('gqa/val_balanced/decode/acc')
  m.append('gqa/testdev_balanced/decode/acc')
  return m
