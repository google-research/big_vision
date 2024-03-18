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
r"""Distill flexible-seqlen ViT on ImageNet-21k from (internal link) B/8.

This config is for reference, we never ran it on public infrastructure.

big_vision.trainers.proj.flexi.distill \
  --config big_vision/configs/proj/flexivit/i21k_distill.py \
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

  pp_label_i21k = f'|onehot({c.num_classes})|keep("image", "prof", "labels")'
  pp_label_i1k = '|onehot(1000, key="{lbl}", key_result="labels")|keep("image", "prof", "labels")'
  c.input.pp = (
      f'decode|inception_crop|flip_lr|copy("image", "prof")'
      f'|resize({c.res})|value_range(-1, 1)'
      f'|resize(224, outkey="prof")|value_range(-1, 1, key="prof")'
      + pp_label_i21k
  )
  pp_eval_both = (
      'decode|copy("image", "prof")|'
      f'|resize_small({c.res//7*8})|central_crop({c.res})|value_range(-1, 1)'
      f'|resize_small(256, key="prof")|central_crop(224, key="prof")|value_range(-1, 1, key="prof")|'
  )
  pp_eval_student = (
      f'decode|resize({c.res//7*8})|central_crop({c.res})|value_range(-1, 1)'
  )
  pp_eval_prof = (
      'decode|resize(256)|central_crop(224)|value_range(-1, 1, outkey="prof")'
  )

  # Aggressive pre-fetching because our models here are small, so we not only
  # can afford it, but we also need it for the smallest models to not be
  # bottle-necked by the input pipeline. Play around with it for -L models tho.
  c.input.prefetch = 8
  c.prefetch_to_device = 4

  c.log_training_steps = 50
  c.ckpt_steps = 1000

  # Model section
  init = 'howto-i21k-B/8'
  c.student_name = 'proj.flexi.vit'
  c.student_init = init
  c.student = dict(variant='B', pool_type='tok', patch_size=(8, 8))

  c.teachers = ['prof']  # You could even add multiple.
  c.prof_name = 'vit'
  c.prof_init = init
  c.prof = dict(variant='B/8', pool_type='tok')

  # Define the model parameters which are flexible:
  c.flexi = dict()
  c.flexi.seqhw = dict(
      # The settings to sample from. Corresponding patch-sizes at 240px:
      # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
      v=(5, 6, 8, 10, 12, 15, 16, 20, 24, 30),
      # The probabilities/weights of them. Default uniform.
      p=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
  )

  # Distillation settings
  c.distance = 'kl'
  c.distance_kw = dict(t=1.0)

  # Optimizer section
  c.optax_name = 'scale_by_adam'
  c.optax = dict(mu_dtype='bfloat16')
  c.grad_clip_norm = 1.0

  c.lr = 1e-4
  c.wd = 1e-5
  c.schedule = dict(warmup_steps=5000, decay_type='cosine')

  c.mixup = dict(p=1.0)

  ####
  # Preparing for evals
  c.evals = {}
  def mksplit(split):
    if c.runlocal:
      return split.split('[')[0] + '[:16]'
    return split

  ####
  # Student evals

  # Evaluations on i21k itself.
  def eval_i21k(s, split):
    return dict(
        type='classification',
        pred=f'student_seqhw={s}',
        data={**c.input.data, 'split': mksplit(split)},
        pp_fn=pp_eval_student + pp_label_i21k,
        loss_name=c.loss,
        log_steps=5000,  # Very fast O(seconds) so it's fine to run it often.
    )

  for s in c.flexi.seqhw.v:
    c.evals[f'student_test{s:02d}'] = eval_i21k(s, 'full[:25_600]')
    c.evals[f'student_val{s:02d}'] = eval_i21k(s, 'full[25_600:51_200]')
    c.evals[f'student_minitrain{s:02d}'] = eval_i21k(s, 'full[51_200:76_800]')

  # Evaluations on ImageNet1k variants by label-mapping.
  def eval_i1k(s, dataset, split, lblmap):
    return dict(
        type='classification_with_labelmap',
        pred=f'student_seqhw={s}',
        data=dict(name=dataset, split=mksplit(split)),
        pp_fn=pp_eval_student + pp_label_i1k.format(lbl='label'),
        loss_name=c.loss,
        log_steps=5000,  # Very fast O(seconds) so it's fine to run it often.
        label_mapping=lblmap,
    )
  for s in c.flexi.seqhw.v:
    c.evals[f'student_i1k_val{s:02d}'] = eval_i1k(s, 'imagenet2012', 'validation', 'i1k_i21k')
    c.evals[f'student_i1k_v2{s:02d}'] = eval_i1k(s, 'imagenet_v2', 'test', 'i1k_i21k')
    c.evals[f'student_i1k_a{s:02d}'] = eval_i1k(s, 'imagenet_a', 'test', 'i1ka_i21k')
    c.evals[f'student_i1k_r{s:02d}'] = eval_i1k(s, 'imagenet_r', 'test', 'i1kr_i21k')
    c.evals[f'student_i1k_real{s:02d}'] = eval_i1k(s, 'imagenet2012_real', 'validation', 'i1k_i21k')
    c.evals[f'student_i1k_real{s:02d}'].pp_fn = pp_eval_student + pp_label_i1k.format(lbl='real_label')
    # TODO: add objectnet.

  ####
  # Teacher evals

  # Evaluations on i21k itself.
  def eval_i21k_t(split):
    return dict(
        type='classification',
        pred='prof',
        data={**c.input.data, 'split': mksplit(split)},
        pp_fn=pp_eval_prof + pp_label_i21k,
        loss_name=c.loss,
        log_steps=5000,  # Very fast O(seconds) so it's fine to run it often.
    )

  c.evals.teacher_test = eval_i21k_t('full[:25_600]')
  c.evals.teacher_val = eval_i21k_t('full[25_600:51_200]')
  c.evals.teacher_minitrain = eval_i21k_t('full[51_200:76_800]')

  # Evaluations on ImageNet1k variants by label-mapping.
  def eval_i1k_t(dataset, split, lblmap):
    return dict(
        type='classification_with_labelmap',
        pred='prof',
        data=dict(name=dataset, split=mksplit(split)),
        pp_fn=pp_eval_prof + pp_label_i1k.format(lbl='label'),
        loss_name=c.loss,
        log_percent=0.5,  # Teacher is fixed, so eval just for plots.
        label_mapping=lblmap,
    )
  c.evals.teacher_i1k_val = eval_i1k_t('imagenet2012', 'validation', 'i1k_i21k')
  c.evals.teacher_i1k_v2 = eval_i1k_t('imagenet_v2', 'test', 'i1k_i21k')
  c.evals.teacher_i1k_a = eval_i1k_t('imagenet_a', 'test', 'i1ka_i21k')
  c.evals.teacher_i1k_r = eval_i1k_t('imagenet_r', 'test', 'i1kr_i21k')
  c.evals.teacher_i1k_real = eval_i1k_t('imagenet2012_real', 'validation', 'i1k_i21k')
  c.evals.teacher_i1k_real.pp_fn = pp_eval_prof + pp_label_i1k.format(lbl='real_label')
  # TODO: add objectnet.

  ####
  # Combined evals

  def get_dist(split, s):
    return dict(
        type='proj.distill.distance',
        pred=f'student_seqhw={s}_prof',
        data=dict(name='imagenet2012', split=mksplit(split)),
        pp_fn=pp_eval_both + '|keep("image", "prof")',
        log_percent=0.05,
        distances=({'kind': 'kl'}, {'kind': 'logsoftmax_euclidean'},
                   {'kind': 'agree', 'k': 1}, {'kind': 'agree', 'k': 5}),
    )
  for s in c.flexi.seqhw.v:
    c.evals[f'dist_minitrain_{s:02d}'] = get_dist('full[51_200:76_800]', s)
    c.evals[f'dist_val_{s:02d}'] = get_dist('full[25_600:51_200]', s)

  # Few-shot evaluators not added for overkill reasons for now.
  return c
