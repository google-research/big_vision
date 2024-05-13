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
r"""PaliGemma transfer to COCO-35L captions.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

LANGUAGES = (
    'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr',
    'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'mi', 'nl', 'no', 'pl',
    'pt', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh',
)

LANGUAGES_XM3600 = (
    'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr',
    'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'mi', 'nl', 'no', 'pl',
    'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh'
)

# A subset for more frequent evals.
LANGUAGES_SUBSET = ('ar', 'bn', 'en', 'id', 'sw', 'tr', 'zh')


def training_data(res, lang=None, text_len=32, crop='rs'):
  """Creates training data config.

  See (internal link)
  You can add more arguments beside `res`, but give them good defaults.

  Args:
    res: The requested image resolution (eg 224)
    lang: language code
    text_len: sequence length
    crop: one of {'ic', 'rc', 'rs'}

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='coco35l',
      split=f'train_{lang}' if lang else '+'.join((f'train_{l}' for l in LANGUAGES)),
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
      'choice_no_replacement(inkey="captions", outkey="suffix")',
      'strfmt("caption {language}", outkey="prefix")',
      combine_and_keep_train(text_len),
  ])
  return c


def _get_eval_pp(res, lang, text_len=32):
  return '|'.join([
      'flatten',
      'decode', f'resize({res})', 'value_range(-1, 1)',
      f'strfmt("caption {lang}", outkey="prefix")',
      combine_and_keep_eval(text_len, keep=('image/id', 'captions')),
  ])


def add_eval(c, res, text_len=32, langs=None, **kw):
  """Captioning evaluator with cider/bleu-4/meteor/rouge/spice metrics."""
  for lang in (langs or LANGUAGES):
    # Frequent evals on a subset of representative languages, final eval on all.
    freq = 0.25 if lang in LANGUAGES_SUBSET else 1.0

    c.evals[f'coco35l/{lang}'] = dict(
        type='proj.paligemma.transfers.coco_caption',
        pred='decode', pred_kw={'max_decode_len': text_len},
        log_percent=freq, skip_first=(freq == 1.0), tokenizer=TOKENIZER,
        data=dict(
            name='coco35l',
            split=f'dev_{lang}',
        ),
        cache='none',
        pp_fn=_get_eval_pp(res, lang, text_len),
    )
    c.evals[f'coco35l/{lang}'].update(kw)


def add_eval_xm(c, res, text_len=32, langs=None, **kw):
  """Captioning evaluator with cider/bleu-4/meteor/rouge/spice metrics."""
  for lang in (langs or LANGUAGES_XM3600):
    # Frequent evals on a subset of representative languages, final eval on all.
    freq = 0.25 if lang in LANGUAGES_SUBSET else 1.0

    c.evals[f'xm3600/{lang}'] = dict(
        type='proj.paligemma.transfers.coco_caption',
        pred='decode', pred_kw={'max_decode_len': text_len},
        log_percent=freq, skip_first=(freq == 1.0), tokenizer=TOKENIZER,
        data=dict(
            name='xm3600',
            split=lang,
        ),
        pp_fn=_get_eval_pp(res, lang, text_len)
    )
    c.evals[f'xm3600/{lang}'].update(kw)


def add_eval_pplx(c, res, text_len=32):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, text_len=text_len)  # Use mostly same settings as training.
  for name, split in [
      ('minitrain', 'train_en[:2%]'),
      ('minival', 'dev_en[:5%]'),
      ('eval', 'dev_en'),
  ]:
    c.evals[f'coco35l/{name}/pplx'] = dict(
        type='proj.paligemma.perplexity', pred='logits',
        key='text', shift_labels=True,
        log_percent=0.05,  # Eval ~20x per run; it's cheap.
        data={**c_train.data, 'split': split},
        pp_fn=c_train.pp,
    )


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', crop='rs', res=224, eval_xm3600=True, beam_size=0)

  c.input = {
      lang: training_data(c.res, lang=lang, crop=c.crop)
      for lang in LANGUAGES
  }
  c.input.data = {lang: 1 for lang in LANGUAGES}
  for k in c.input.data:
    c.input[k].shuffle_buffer_size = 10_000

  c.total_examples = 566_435  # We need to go a looot longer here.
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
  if c.eval_xm3600:
    add_eval_xm(c, c.res, batch_size=1024, **decode_kw)

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
  c.log_training_steps = 50
  c.ckpt_steps = 1_000
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  # Update configs for quicker local runs and avoid swapping.
  if c.mode in ('runlocal', 'mock'):
    # c.input.shuffle_buffer_size = None
    for ev in c.evals.values():
      ev.data.split = ev.data.split.split('[')[0] + '[:16]'

  if c.mode == 'runlocal':
    c.log_training_steps = 1
    c.input.batch_size = 2

  c.seed = 0
  return c


def sweep_hyper(add):
  """sweep over hyper-params."""
  for lr in (1e-5, 3e-6, 1e-6):
    for wd in (0.0, 0.1*lr):
      for ep in (1, 3, 5, 10, 20):
        # One language COCO is 566_435 examples (5 captions, 100k examples).
        add(lr=lr, wd=wd, total_examples=ep * 566_435, **bvcc.arg(res=224))


def sweep_best(add, arg=None):
  """Train with best hyper-params."""
  c = bvcc.parse_arg(arg, eval_xm3600=True)
  ep = 566_435
  add(lr=1e-5, wd=1e-6, total_examples=5 * ep, **bvcc.arg(res=224, **c))
  add(lr=1e-5, wd=1e-6, total_examples=5 * ep, **bvcc.arg(res=448, **c))


sweep = sweep_best  # Choose which sweep to run.


def metrics(arg=None):  # pylint: disable=unused-argument
  c = bvcc.parse_arg(arg, eval_xm3600=True)
  m = [('epoch', f'coco35l/{lang}/cider') for lang in LANGUAGES]
  if c.eval_xm3600:
    for lang in LANGUAGES:
      m.append(('epoch', f'xm3600/{lang}/cider'))
  return m
