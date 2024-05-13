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
r"""PaliGemma transfer to chartqa.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

_DATASETS = ('chartqa/human', 'chartqa/augmented')
# We use the true dataset sizes from https://arxiv.org/pdf/2203.10244.pdf.
_WEIGHTS = (7_398, 20_901)


def training_data(res, *, final_split=False, text_len=48):
  """Creates training data config.

  See (internal link)
  You can add more arguments beside `res`, but give them good defaults.

  Args:
    res: The requested image resolution (eg 224).
    final_split: Train on all train+val data.
    text_len: sequence length.

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  pp = '|'.join([
      f'decode|resize({res}, antialias=True)|value_range(-1, 1)',
      'copy(inkey="question", outkey="prefix")',
      'copy(inkey="answer", outkey="suffix")',
      combine_and_keep_train(text_len),
  ])
  c.data = {ds: weight for ds, weight in zip(_DATASETS, _WEIGHTS)}
  for ds in c.data:
    c[ds] = dict(
        shuffle_buffer_size=50_000,
        pp=pp,
        data=dict(
            name=ds,
            split='train+val' if final_split else 'train',
        ),
    )
  return c


def add_eval(c, res, text_len=48, **kw):
  """Add eval configs."""
  c_train = training_data(res, final_split=True, text_len=text_len)

  pp_eval = '|'.join([
      f'decode|resize({res}, antialias=True)|value_range(-1, 1)',
      'copy(inkey="question", outkey="prefix")',
      combine_and_keep_eval(text_len, keep=('answer', 'question_id')),
  ])

  for name, split in [
      ('minitrain', 'train[:5%]'),
      ('minival', 'val'),
      ('eval', 'test'),
  ]:
    for ds in _DATASETS:
      c.evals[f'{ds}/{name}'] = dict(
          type='proj.paligemma.transfers.chartqa',
          pred='decode', pred_kw={'max_decode_len': text_len},
          to_lower=True,
          outfile=f'{{workdir}}/{ds.replace("/", "_")}_{name}.json',
          data={**c_train[ds].data, 'split': split},
          log_percent=0.1, tokenizer=TOKENIZER, pp_fn=pp_eval)
      c.evals[f'{ds}/{name}'].update(kw)


def add_eval_pplx(c, res, text_len=48):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, final_split=True, text_len=text_len)  # Use mostly same settings as training.
  for name, split in [
      ('minitrain', 'train[:5%]'),  # To gauge memorization.
      ('minival', 'val'),  # To tune hparams.
      ('eval', 'test'),  # To compute final publishable scores.
  ]:
    for ds in _DATASETS:
      c.evals[f'{ds}/{name}/pplx'] = dict(
          type='proj.paligemma.perplexity', pred='logits',
          key='text', shift_labels=True,
          log_percent=0.05,  # Eval ~20x per run; it's cheap.
          data={**c_train[ds].data, 'split': split},
          pp_fn=c_train[ds].pp,
      )


def sweep_best(add, arg=None):
  """Train with best hyper-params."""
  c = bvcc.parse_arg(arg, final_split=False)
  # TODO: Update once latest numbers are in and have only 1 setup.
  # Based on (internal link) (relaxed_accuracy).
  add(lr=1e-5, wd=1e-6, total_epochs=30, **bvcc.arg(res=224, **c))
  # Based on sweep (internal link) and on (internal link) (relaxed_accuracy).
  add(lr=1e-5, wd=1e-6, total_epochs=30, **bvcc.arg(res=448, **c))
  # Based on (internal link) (relaxed_accuracy).
  # Not better: add(lr=1e-5, wd=1e-6, total_epochs=30, **bvcc.arg(res=896, **c))


sweep = sweep_best  # Choose which sweep to run.


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', res=896, final_split=False)

  c.input = training_data(c.res, final_split=c.final_split)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 30
  c.input.batch_size = 256
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 1e-5
  c.wd = 1e-6
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.2
  c.schedule = dict(decay_type='cosine', warmup_percent=0.05)

  # Add evaluators.
  c.evals = {}
  add_eval(c, c.res, batch_size=1024)
  add_eval_pplx(c, c.res)

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.1)
  c.model_init = f'pt_{c.res}'

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  # These probably do not need any change/tuning
  c.log_training_steps = 50
  c.ckpt_steps = 1_000
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  # Update configs for quicker local runs and avoid swapping.
  if c.mode in ('runlocal', 'mock'):
    for ds in _DATASETS:
      c.input[ds].shuffle_buffer_size = None
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
    for ds in _DATASETS:
      m.append(f'{ds}/{split}/relaxed_acc')
      m.append(f'{ds}/{split}/pplx/avg')
  return m
