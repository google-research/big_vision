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
r"""PaliGemma transfer to COCO captions.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER


def training_data(res, *, final_split, text_len=32, crop='rs'):
  """Creates training data config.

  See (internal link)
  You can add more arguments beside `res`, but give them good defaults.

  Args:
    res: The requested image resolution (eg 224).
    final_split: Train on all train+restval data or train[:98%]+restval.
    text_len: sequence length
    crop: one of {'ic', 'rc', 'rs'}

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='coco_captions',
      split='train+restval' if final_split else 'train[:98%]+restval',
  )

  if crop == 'ic':
    crop = f'inception_crop({res}, area_min=50)'
  elif crop == 'rc':
    crop = f'resize_small({res*8//7})|random_crop({res})'
  elif crop == 'rs':
    crop = f'resize({res})'
  else:
    raise ValueError(f'Unknown crop: {crop}')

  c.pp = '|'.join([
      'flatten',
      'decode', crop, 'value_range(-1, 1)',
      'choice_no_replacement(inkey="captions/text", outkey="suffix")',
      'strfmt("caption en", outkey="prefix")',
      combine_and_keep_train(text_len),
  ])
  return c


def add_eval(c, res, text_len=32, **kw):
  """Captioning evaluator with cider/bleu-4/meteor/rouge/spice metrics."""
  # Input eval pp without ground truth text and random crop.
  pp_eval = '|'.join([
      'decode', f'resize({res})', 'value_range(-1, 1)',
      'strfmt("caption en", outkey="prefix")',
      combine_and_keep_eval(text_len, keep=('image/id', 'captions')),
  ])

  for name, split in [
      ('minitrain', 'train[:2%]'),
      ('minival', 'train[-2%:]'),
      ('eval', 'val'),
  ]:
    c.evals[f'cococap/{name}'] = dict(
        type='proj.paligemma.transfers.coco_caption',
        pred='decode', pred_kw={'max_decode_len': text_len},
        log_percent=0.1, tokenizer=TOKENIZER,
        data={'name': 'coco_captions', 'split': split},
        pp_fn='|'.join([
            'flatten', 'copy("captions/text", "captions")',  # GT for evaluator.
            pp_eval,
        ]),
    )
    c.evals[f'cococap/{name}'].update(kw)

  c.evals['nocaps/eval'] = dict(
      type='proj.paligemma.transfers.coco_caption',
      pred='decode', pred_kw={'max_decode_len': text_len},
      log_percent=0.1, tokenizer=TOKENIZER,
      data={'name': 'nocaps', 'split': 'val'},
      pp_fn='|'.join([
          'copy("texts", "captions")',  # GT for evaluator.
          pp_eval,
      ]),
  )
  c.evals['nocaps/eval'].update(kw)


def add_eval_pplx(c, res, text_len=32):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, final_split=True, text_len=text_len, crop='rs')  # Use mostly same settings as training.
  for name, split in [
      ('minitrain', 'train[:2%]'),
      ('minival', 'train[-2%:]'),
      ('eval', 'val'),
  ]:
    c.evals[f'cococap/{name}/pplx'] = dict(
        type='proj.paligemma.perplexity', pred='logits',
        key='text', shift_labels=True,
        log_percent=0.05,  # Eval ~20x per run; it's cheap.
        data={**c_train.data, 'split': split},
        pp_fn=c_train.pp,
    )


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', crop='rs', res=224, beam_size=2, final_split=False)

  c.input = training_data(c.res, final_split=c.final_split, crop=c.crop)

  c.total_epochs = 5  # One epoch of captions (each image has 5 captions).
  c.input.batch_size = 256
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 1e-4
  c.wd = 0.0
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0
  c.schedule = dict(decay_type='cosine', warmup_percent=0.05)

  # Add evaluators.
  c.evals = {}
  add_eval_pplx(c, c.res)

  if c.beam_size:
    decode_kw = {'pred': 'beam_decode', 'pred_kw': {'beam_size': c.beam_size}}
  else:
    decode_kw = {}

  add_eval(c, c.res, batch_size=1024, **decode_kw)

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


def sweep_best(add, arg=None):
  """Train with best hyper-params."""
  c = bvcc.parse_arg(arg, final_split=False)
  # Note: wd=0.0 works as good.
  add(lr=1e-5, wd=1e-6, total_epochs=5, **bvcc.arg(res=224, **c))
  add(lr=1e-5, wd=1e-6, total_epochs=5, **bvcc.arg(res=448, **c))


sweep = sweep_best  # Choose which sweep to run.


def metrics(arg=None):  # pylint: disable=unused-argument
  m = ['training_loss']
  for split in ('eval', 'minival', 'minitrain'):
    m.append(('epoch', f'cococap/{split}/cider'))
    m.append(('epoch', f'cococap/{split}/pplx/avg'))
  return m
