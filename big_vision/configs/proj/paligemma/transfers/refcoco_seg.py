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
r"""PaliGemma transfer to RefCOCO (with segmentation).
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER


def training_data(res, text_len=48, crop='rs'):
  """Creates training data config.

  See (internal link)
  You can add more arguments beside `res`, but give them good defaults.

  Args:
    res: The requested image resolution (eg 224)
    text_len: sequence length
    crop: What way to do random cropping to get to res.

  Returns:
    The ConfigDict for the input section.
  """
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='ref_coco_bv/refcocox_combined:1.4.0',
      split='train',
  )

  if crop == 'rs':
    crop_ops = f'resize({res})'
  elif crop == 'zic_mild':
    crop_ops = '|'.join([
        'zoomout(max_f=1.5, key="image", bboxkey="objects/bbox", auxkeys=["objects/mask"])',
        f'inception_box(area=(0.1,1.0), aspect=({3/4},{4/3}), min_obj_cover=1.0, bboxkey="objects/bbox")',
        'box_crop_bbox',
        'box_crop_img(key="objects/mask")',
        'box_crop_img(key="image")',
        f'resize({res})',
    ])
  else:
    raise ValueError(crop)

  c.pp = '|'.join([
      'flatten',
      'choice_no_replacement(key=["objects/mask", "objects/bbox", "objects/refs/sentence"])',
      'choice(key=["objects/refs/sentence"])',
      'decode',
      crop_ops,
      'value_range(-1, 1)',
      'refcoco_mask2str',
      combine_and_keep_train(text_len),
  ])
  return c


def add_eval(c, res, text_len=48, **kw):
  """Segmentation evaluator computing mIoU."""
  # NOTE: we verified that squeezing to square at eval time is no worse than
  # padding with black borders. The actual evaluation is still done in original
  # full resolution of the mask, of course.
  pp_eval_squeeze = '|'.join([
      'flatten',
      # (choice simply removes a dimension since it's already flattened)
      'choice(key=["objects/mask", "objects/bbox", "objects/refs/sentence"])',
      'choice(key=["objects/refs/sentence"], outkey="prefix")',
      # 'refcoco_mask2str',  # TODO: b/lbeyer - also eval decoded GT mask?
      f'decode|resize({res})|value_range(-1, 1)',
      combine_and_keep_eval(text_len, keep=('objects/mask', 'objects/bbox', 'width', 'height')),
  ])

  for freq, name, ds_name, split in [
      (0.2, 'refcoco/val', 'ref_coco_bv/refcoco_unc:1.4.0', 'validation_flat'),
      (1.0, 'refcoco/testA', 'ref_coco_bv/refcoco_unc:1.4.0', 'testA_flat'),
      (1.0, 'refcoco/testB', 'ref_coco_bv/refcoco_unc:1.4.0', 'testB_flat'),
      (1.0, 'refcocop/val', 'ref_coco_bv/refcocoplus_unc:1.4.0', 'validation_flat'),
      (1.0, 'refcocop/testA', 'ref_coco_bv/refcocoplus_unc:1.4.0', 'testA_flat'),
      (1.0, 'refcocop/testB', 'ref_coco_bv/refcocoplus_unc:1.4.0', 'testB_flat'),
      (1.0, 'refcocog/val', 'ref_coco_bv/refcocog_umd:1.4.0', 'validation_flat'),
      (1.0, 'refcocog/test', 'ref_coco_bv/refcocog_umd:1.4.0', 'test_flat'),
  ]:
    c.evals[f'seg/{name}'] = dict(
        type='proj.paligemma.transfers.segmentation',
        pred='decode', pred_kw={'max_decode_len': text_len},
        data={'name': ds_name, 'split': split},
        log_percent=freq, skip_first=freq == 1,
        tokenizer=TOKENIZER, pp_fn=pp_eval_squeeze)
    c.evals[f'seg/{name}'].update(kw)


def add_eval_pplx(c, res, text_len=48):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, text_len)  # Use mostly same settings as training.
  for name, split in [
      ('minitrain', 'train[:5%]'),  # 1_220
      ('val', 'validation'),  # 2_738
  ]:
    c.evals[f'refcoco_seg/{name}/pplx'] = dict(
        type='proj.paligemma.perplexity', pred='logits',
        key='text', shift_labels=True,
        log_percent=0.05,  # Eval ~20x per run; it's cheap.
        data={**c_train.data, 'split': split},
        pp_fn=c_train.pp,
    )


def sweep_best(add, arg=None):  # pylint: disable=unused-argument
  """Train with best hyper-params per resolution."""
  # Based on (internal link)
  add(**bvcc.arg(res=224), lr=3e-5, total_epochs=100, label_smoothing=0.3,
      **{'model.llm.dropout': 0.1, 'input.batch_size': 256})
  add(**bvcc.arg(res=448), lr=1e-5, total_epochs=100, label_smoothing=0.3,
      **{'model.llm.dropout': 0.0, 'input.batch_size': 256})
  # Takes 2d on 16 TPUv5e, gives overall +0.5-+1 over 448.
  add(**bvcc.arg(res=896), lr=1e-5, total_epochs=100, label_smoothing=0.3,
      **{'model.llm.dropout': 0.0, 'input.batch_size': 64})


sweep = sweep_best  # Choose which sweep to run.


def get_config(arg=None):
  """Default "reasonably good" training config, gets about 75/67/70."""
  c = bvcc.parse_arg(arg, mode='xm', res=448, crop='rs')

  c.input = training_data(c.res, crop=c.crop)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 5
  c.input.batch_size = 64
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 1e-5
  c.wd = 0.0
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.3
  c.schedule = dict(decay_type='cosine', warmup_percent=0.05)

  # Add evaluators.
  c.evals = {}
  add_eval(c, c.res, batch_size=256)
  add_eval_pplx(c, c.res)

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True, dropout=0.0)
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
      'ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops',
      'proj.paligemma.segmentation',
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
  for thing in ('miou', 'boxacc/0.5', 'invalid'):
    m.append(f'seg/refcoco/val/{thing}')
  for split in ('/testA', '/testB', 'p/val', 'p/testA', 'p/testB', 'g/val', 'g/test'):
    m.append(f'seg/refcoco{split}/miou')
  return m
