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
r"""PaliGemma transfer to science_qa.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER


def training_data(res, *, final_split, text_len=512, qfmt='QCM', afmt='A'):
  """Creates training data config.

  See
  (internal link)
  You can add more arguments beside `res`, but give them good defaults.
  implemented based on :
  https://github.com/lupantech/ScienceQA/blob/main/models/base_prompt.py
  default prompt format baseline: QCM -> A (see paper)
  Args:
    res: The requested image resolution (eg 224).
    final_split: Train on all train+val data.
    text_len: sequence length.
    qfmt: see config_prompt_format.
    afmt: see config_prompt_format.

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='science_qa',
      split='train+val' if final_split else 'train',
  )
  qfmt, afmt = config_prompt_format(qfmt, afmt)
  c.pp = '|'.join([
      # Read and prepare the image by just resizing it:
      f'decode|resize({res})|value_range(-1, 1)',
      'drop("indexed_choices","indexed_answer")',
      "sci_qa_choices_shuffle(choice_str_inkey='choices', ans_inkey='answer')",
      f'strfmt("{qfmt}", outkey="prefix")',
      f'strfmt("{afmt}", outkey="suffix")',
      combine_and_keep_train(text_len),
  ])
  return c


def config_prompt_format(qfmt='QCM', afmt='A'):
  """Configure prompt format, default: QCM -> A.

  See https://github.com/lupantech/ScienceQA/blob/main/models/base_prompt.py
  Args:
    qfmt: input prompt format -> the question text (Q), the context text
      (C), and multiple options (M)
    afmt: out prompt format -> A = answer, AE = answer with
      explanation, ALE = answer with lecture and explanation.

  Returns:
    prompt format string for training data config to digest
  """
  # TODO: b/keranrong - The \\nAnswer: is useless for our model.
  if qfmt == 'simple':
    qfmt = '{question}\\nOptions: {indexed_choices}'
  elif qfmt == 'QM':
    qfmt = 'Question: {question}\\nOptions: {indexed_choices}\\nAnswer:'
  elif qfmt == 'CQM':
    qfmt = 'Context: {hint}\\nQuestion: {question}\\nOptions: {indexed_choices}\\nAnswer:'
  elif qfmt == 'QCM':
    qfmt = 'Question: {question}\\nContext: {hint}\\nOptions: {indexed_choices}\\nAnswer:'
  else:
    raise ValueError(qfmt)

  if afmt == 'simple':
    afmt = '{indexed_answer}'
  elif afmt == 'A':
    afmt = 'The answer is {indexed_answer}.'
  elif afmt == 'AL':
    afmt = 'The answer is {indexed_answer}. BECAUSE: {solution}'
  elif afmt == 'AE':
    afmt = 'The answer is {indexed_answer}. BECAUSE: {lecture}'
  elif afmt == 'ALE':
    afmt = 'The answer is {indexed_answer}. BECAUSE: {lecture} {solution}'
  else:
    raise ValueError(afmt)

  return qfmt, afmt


def add_eval(c, res, text_len=512, qfmt='QCM', afmt='A', **kw):
  """Science QA evaluators."""
  prefix, suffix = config_prompt_format(qfmt, afmt)
  pp = '|'.join([
      f'decode|resize({res})|value_range(-1, 1)',
      f'strfmt("{prefix}", outkey="prefix")',
      f'strfmt("{suffix}", outkey="answer")',
      'copy(inkey="_id",outkey="question_id")',
      combine_and_keep_eval(text_len, keep=('answer', 'question_id')),
  ])

  for name, split in [
      ('minitrain', 'train[:5%]'),  # To gauge memorization.
      ('minival', 'val'),  # To tune hparams.
      ('eval', 'test'),
  ]:
    c.evals[f'science_qa/{name}'] = dict(
        type='proj.paligemma.transfers.science_qa',
        pred='decode', pred_kw={'max_decode_len': text_len},
        outfile=f'{{workdir}}/science_{name}.json',
        data={**training_data(res, final_split=True, text_len=text_len, qfmt=qfmt, afmt=afmt).data, 'split': split},
        log_percent=1/8, tokenizer=TOKENIZER, pp_fn=pp)
    c.evals[f'science_qa/{name}'].update(kw)


def add_eval_pplx(c, res, text_len=512, qfmt='QCM', afmt='A'):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, final_split=True, text_len=text_len, qfmt=qfmt, afmt=afmt)  # Use mostly same settings as training.
  for name, split in [
      ('minitrain', 'train[:5%]'),  # To gauge memorization.
      ('minival', 'val'),  # To tune hparams.
  ]:
    c.evals[f'science_qa/{name}/pplx'] = dict(
        type='proj.paligemma.perplexity', pred='logits',
        key='text', shift_labels=True,
        log_percent=0.05,  # Eval ~20x per run; it's cheap.
        data={**c_train.data, 'split': split},
        pp_fn=c_train.pp,
    )


def sweep_best(add, arg=None):
  """Train with best hyper-params."""
  c = bvcc.parse_arg(arg, final_split=False)
  # Based on sweep xids/98045006 (science_qa/eval/acc).
  # TODO: b/keranrong - try getting rid of freezing
  add(lr=1e-5, wd=0, **bvcc.arg(freeze_vit=True, res=224, **c))
  add(lr=1e-5, wd=0, **bvcc.arg(freeze_vit=True, res=448, **c))


sweep = sweep_best  # Choose which sweep to run.


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', res=448, final_split=False, freeze_vit=False, freeze_llm=False, qfmt='QCM', afmt='A')

  c.input = training_data(c.res, final_split=c.final_split, qfmt=c.qfmt, afmt=c.afmt)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 20
  c.input.batch_size = 128
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 5e-5
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
  add_eval(c, c.res, qfmt=c.qfmt, afmt=c.afmt, batch_size=1024)
  add_eval_pplx(c, c.res, qfmt=c.qfmt, afmt=c.afmt)

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
  c.pp_modules = [
      'ops_general',
      'ops_image',
      'ops_text',
      'proj.paligemma.ops',
      'proj.paligemma.sciqa_ops',
  ]

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
    m.append(f'science_qa/{split}/pplx/avg')
    m.append(f'science_qa/{split}/acc')
  return m
