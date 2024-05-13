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
r"""PaliGemma transfer to NLVR2 captions, including evaluation on MaRVL.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

LANGS = ('id', 'sw', 'ta', 'tr', 'zh')


def training_data(res, *, final_split, text_len=64):
  """Creates training data config.

  See (internal link)
  You can add more arguments beside `res`, but give them good defaults.

  Args:
    res: The requested image resolution (eg 224).
    final_split: Train on all train+dev data.
    text_len: sequence length.

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='nlvr2',
      split='train+dev' if final_split else 'train',
  )

  num_frames = 2
  c.pp = '|'.join([
      f'resize({res}, key="image_left")|resize({res}, key="image_right")',
      'stack_images(inkeys=["image_left", "image_right"], outkey="image")',
      'value_range(-1, 1)',
      f'video_ensure_shape("image", {(num_frames, res, res, 3)})',
      'strfmt("answer en {sentence}", outkey="prefix")',
      'copy(inkey="label", outkey="suffix")',
      combine_and_keep_train(text_len),
  ])
  return c


def add_eval(c, res, text_len=64, **kw):
  """QA evaluator."""
  # Input eval pp without ground truth text and random crop.
  num_frames = 2
  pp_eval = '|'.join([
      f'resize({res}, key="image_left")|resize({res}, key="image_right")',
      'stack_images(inkeys=["image_left", "image_right"], outkey="image")',
      'value_range(-1, 1)',
      f'video_ensure_shape("image", {(num_frames, res, res, 3)})',
      'strfmt("answer en {sentence}", outkey="prefix")',
      'copy(inkey="label", outkey="answer")',
      'copy(inkey="example_id", outkey="question_id")',
      combine_and_keep_eval(text_len, keep=('answer', 'question_id')),
  ])

  for name, split in [
      ('minitrain', 'train[:10000]'),
      ('dev', 'dev'),
      ('test', 'test'),
  ]:
    c.evals[f'nlvr2/{name}'] = dict(
        type='proj.paligemma.transfers.vqa',
        pred='decode', pred_kw={'max_decode_len': text_len},
        outfile=f'{{workdir}}/nlvr2_{name}.json',
        data={**training_data(res, final_split=True, text_len=text_len).data, 'split': split},
        log_percent=0.1, tokenizer=TOKENIZER, pp_fn=pp_eval)
    c.evals[f'nlvr2/{name}'].update(kw)

  for lang in LANGS:
    c.evals[f'marvl/test_{lang}'] = dict(
        type='proj.paligemma.transfers.vqa',
        pred='decode', pred_kw={'max_decode_len': text_len},
        outfile=f'{{workdir}}/marvl_test_{lang}.json',
        data=dict(
            name='marvl',
            split=f'test_{lang}',
        ),
        log_percent=0.1, tokenizer=TOKENIZER, pp_fn=pp_eval)
    c.evals[f'marvl/test_{lang}'].update(kw)


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', res=224, final_split=False)

  c.input = training_data(c.res, final_split=c.final_split)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 3
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
  add_eval(c, c.res, batch_size=1024)

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
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops',
                  'proj.paligemma.video']

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
  add(lr=1e-5, wd=1e-6, total_epochs=3, **bvcc.arg(res=224, **c))
  add(lr=3e-6, wd=3e-7, total_epochs=10, **bvcc.arg(res=448, **c))


sweep = sweep_best


def metrics(arg=None):  # pylint: disable=unused-argument
  m = ['training_loss']
  for split in ('minitrain', 'minival', 'dev', 'test'):
    m.append(f'nlvr2/{split}/acc')
  for lang in LANGS:
    m.append(f'marvl/test_{lang}/acc')
  return m
