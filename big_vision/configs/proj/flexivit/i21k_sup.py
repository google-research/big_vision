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
r"""Pre-training flexible-seqlen ViT on ImageNet-21k following (internal link).

This config is for reference, we never ran it on public infrastructure.

big_vision.trainers.proj.flexi.train \
  --config big_vision/configs/proj/flexivit/i21k_sup.py \
  --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
  --config.total_epochs 90
"""

import big_vision.configs.common as bvcc


def get_config(arg=None):
  """Config for training."""
  # 240px is nice because it's divisible by
  # [240, 120, 80, 60, 48, 40, 30, 24, 20, 16, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1]
  c = bvcc.parse_arg(arg, runlocal=False, res=240)

  c.seed = 0
  c.total_epochs = 90
  c.num_classes = 21843
  c.init_head_bias = -10.0
  c.loss = 'sigmoid_xent'

  c.input = dict()
  c.input.data = dict(
      name='imagenet21k',
      split='full[51200:]',
  )
  c.input.batch_size = 4096 if not c.runlocal else 8
  c.input.shuffle_buffer_size = 250_000 if not c.runlocal else 25

  pp_common = '|value_range(-1, 1)|onehot({onehot_args})|keep("image", "labels")'
  pp_common_i21k = pp_common.format(onehot_args=f'{c.num_classes}')
  pp_common_i1k = pp_common.format(onehot_args='1000, key="{lbl}", key_result="labels"')
  c.input.pp = f'decode_jpeg_and_inception_crop({c.res})|flip_lr|randaug(2,10)' + pp_common_i21k
  def pp_eval(res=c.res):
    return f'decode|resize_small({res//7*8})|central_crop({res})'

  # To continue using the near-defunct randaug op.
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'archive.randaug']

  # Aggressive pre-fetching because our models here are small, so we not only
  # can afford it, but we also need it for the smallest models to not be
  # bottle-necked by the input pipeline. Play around with it for -L models tho.
  c.input.prefetch = 8
  c.prefetch_to_device = 4

  c.log_training_steps = 50
  c.ckpt_steps = 1000

  # Model section
  c.model_name = 'proj.flexi.vit'
  c.model = dict(
      variant='B',
      pool_type='tok',
      posemb='learn',
      # patch_size=(32, 32),
      patch_size=(8, 8),
      posemb_size=(7, 7),
      seqhw=None,  # Dynamic!
  )

  # Define the model parameters which are flexible:
  c.flexi = dict()
  c.flexi.seqhw = dict(
      # The settings to sample from. Corresponding patch-sizes at 240px:
      # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
      v=(5, 6, 8, 10, 12, 15, 16, 20, 24, 30),
      # The probabilities/weights of them. Default uniform.
      p=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
  )

  # Optimizer section
  c.optax_name = 'scale_by_adam'
  c.optax = dict(mu_dtype='bfloat16')
  c.grad_clip_norm = 1.0

  c.lr = 0.001
  c.wd = 0.0001
  c.schedule = dict(warmup_steps=10_000, decay_type='cosine')

  c.mixup = dict(p=0.2, fold_in=None)

  def mksplit(split):
    if c.runlocal:
      return split.split('[')[0] + '[:16]'
    return split

  # Evaluations on i21k itself.
  def eval_i21k(s, split):
    return dict(
        type='classification',
        pred=f'predict_seqhw={s}',
        data={**c.input.data, 'split': mksplit(split)},
        pp_fn=pp_eval() + pp_common_i21k,
        loss_name=c.loss,
        log_steps=5000,  # Very fast O(seconds) so it's fine to run it often.
    )

  c.evals = {}
  for s in c.flexi.seqhw.v:
    c.evals[f'test{s:02d}'] = eval_i21k(s, 'full[:25_600]')
    c.evals[f'val{s:02d}'] = eval_i21k(s, 'full[25_600:51_200]')
    c.evals[f'train{s:02d}'] = eval_i21k(s, 'full[51_200:76_800]')

  # Evaluations on ImageNet1k variants by label-mapping.
  def eval_i1k(s, dataset, split, lblmap):
    return dict(
        type='classification_with_labelmap',
        pred=f'predict_seqhw={s}',
        data=dict(name=dataset, split=mksplit(split)),
        pp_fn=pp_eval() + pp_common_i1k.format(lbl='label'),
        loss_name=c.loss,
        log_steps=5000,  # Very fast O(seconds) so it's fine to run it often.
        label_mapping=lblmap,
    )
  for s in c.flexi.seqhw.v:
    c.evals[f'i1k_val{s:02d}'] = eval_i1k(s, 'imagenet2012', 'validation', 'i1k_i21k')
    c.evals[f'i1k_v2{s:02d}'] = eval_i1k(s, 'imagenet_v2', 'test', 'i1k_i21k')
    c.evals[f'i1k_a{s:02d}'] = eval_i1k(s, 'imagenet_a', 'test', 'i1ka_i21k')
    c.evals[f'i1k_r{s:02d}'] = eval_i1k(s, 'imagenet_r', 'test', 'i1kr_i21k')
    c.evals[f'i1k_real{s:02d}'] = eval_i1k(s, 'imagenet2012_real', 'validation', 'i1k_i21k')
    c.evals[f'i1k_real{s:02d}'].pp_fn = pp_eval() + pp_common_i1k.format(lbl='real_label')
    # TODO: add objectnet.

  # Few-shot evaluators not added for overkill reasons for now.
  return c
