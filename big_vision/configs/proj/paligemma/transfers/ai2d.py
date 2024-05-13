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
r"""PaliGemma transfer to AI2D.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

PREFIX = 'answer en '
PROMPT = 'choose from:'
PROMPT_SEP = ' \t '


def training_data(res, final_split, text_len=128):
  """Creates training data config.

  See (internal link)
  You can add more arguments beside `res`, but give them good defaults.

  Args:
    res: The requested image resolution (eg 224).
    final_split: whether to use all train data.
    text_len: sequence length

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='ai2d',
      # 12k training examples.
      split='train' if final_split else 'train[:-1024]',
  )
  c.pp = '|'.join([
      f'decode|resize({res}, antialias=True)|value_range(-1, 1)',
      f'strjoin("{PROMPT_SEP}", inkey="possible_answers", outkey="ansstr")',
      f'strfmt("{PREFIX} {{question}} {PROMPT} {{ansstr}}", outkey="prefix")',
      'copy(inkey="answer", outkey="suffix")',
      combine_and_keep_train(text_len),
  ])
  return c


def add_eval(c, res, text_len=128, **kw):
  """AI2D evaluators."""
  pp = '|'.join([
      f'decode|resize({res})|value_range(-1, 1)',
      f'strjoin("{PROMPT_SEP}", inkey="possible_answers", outkey="ansstr")',
      f'strfmt("{PREFIX} {{question}} {PROMPT} {{ansstr}}", outkey="prefix")',
      'copy(inkey="id",outkey="question_id")',
      combine_and_keep_eval(text_len, keep=('answer', 'question_id')),
  ])

  for name, split in [
      ('minitrain', 'train[:1024]'),  # To gauge memorization.
      ('minival', 'train[-1024:]'),   # To tune hparams.
      ('eval', 'test'),               # To compute final publishable scores.
  ]:
    c.evals[f'ai2d/{name}'] = dict(
        type='proj.paligemma.transfers.vqa',
        pred='decode', pred_kw={'max_decode_len': text_len},
        outfile=f'{{workdir}}/ai2d_{name}.json',
        to_lower=False,  # Model sees options in prompt and can match the case.
        data={**training_data(res, True, text_len).data, 'split': split},
        log_percent=1/8, tokenizer=TOKENIZER, pp_fn=pp)
    c.evals[f'ai2d/{name}'].update(kw)


def add_eval_pplx(c, res, text_len=128):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, True, text_len)  # Use mostly same settings as training.

  for name, split in [
      ('minitrain', 'train[:1024]'),  # To gauge memorization.
      ('minival', 'train[-1024:]'),   # To tune hparams.
      ('eval', 'test'),               # To compute final publishable scores.
  ]:
    c.evals[f'ai2d/{name}/pplx'] = dict(
        type='proj.paligemma.perplexity', pred='logits',
        key='text', shift_labels=True,
        log_percent=1/8,
        data={**c_train.data, 'split': split},
        pp_fn=c_train.pp,
    )


def sweep_best(add, arg=None):
  """Train with best hyper-params."""
  c = bvcc.parse_arg(arg, final_split=False)
  add(lr=1e-5, wd=1e-6, total_epochs=10, **bvcc.arg(res=224, **c))
  add(lr=1e-5, wd=1e-6, total_epochs=10, **bvcc.arg(res=448, **c))
  # 896 was not better than 448 ((internal link)).


sweep = sweep_best  # Choose which sweep to run.


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', res=224, final_split=False)

  c.input = training_data(c.res, final_split=c.final_split)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 10
  c.input.batch_size = 256
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 1e-5
  c.wd = 1e-5 * 0.1
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0
  c.schedule = dict(decay_type='cosine', warmup_percent=0.05)

  # Add evaluators.
  c.evals = {}
  add_eval(c, c.res, batch_size=1024)
  add_eval_pplx(c, c.res)

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


def metrics(arg=None):  # pylint: disable=unused-argument
  m = ['training_loss']
  for split in ('eval', 'minival', 'minitrain'):
    m.append(f'ai2d/{split}/pplx/avg')
    m.append(f'ai2d/{split}/acc')
  return m
