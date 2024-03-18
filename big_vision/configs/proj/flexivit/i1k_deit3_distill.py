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
r"""Distillation of ViT models into FlexiViT on ImageNet1k.

Run training of the -S variant for 90ep:

big_vision.trainers.proj.flexi.distill \
  --config big_vision/configs/proj/flexivit/i1k_deit3_distill.py \
  --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
  --config.total_epochs 90 --config.variant S

Logdir for one reproduction run:
  - gs://big_vision/flexivit/deit3_i1k_s_90ep_12-15_2254

Timing on Cloud:
  - S on v3-32: Walltime:10h16m (4h39m eval)

Note that we did not optimize the input for Cloud,
with tuned caching and prefetching, we should be able to get:
  - S on v3-32: Walltime: ~6h30m (~1h30m eval)
  - B on v3-32: Walltime: ~16h00m (~2h30m eval)
"""

import big_vision.configs.common as bvcc


def get_config(arg=None):
  """Config for distilling ViT on ImageNet1k."""
  c = bvcc.parse_arg(arg, runlocal=False, res=240)

  c.seed = 0
  c.total_epochs = 90
  c.num_classes = 1000
  c.loss = 'softmax_xent'

  c.input = {}
  c.input.data = dict(
      name='imagenet2012',
      split='train[:99%]',
  )
  c.input.batch_size = 1024 if not c.runlocal else 8
  c.input.cache_raw = False  # Needs up to 120GB of RAM!
  c.input.shuffle_buffer_size = 250_000 if not c.runlocal else 10

  c.log_training_steps = 50
  c.ckpt_steps = 1000

  # Model section
  c.variant = 'B'
  init = bvcc.format_str('deit3_{variant}_384_1k', c)
  c.student_name = 'proj.flexi.vit'
  c.student_init = init
  c.student = dict(variant=c.get_ref('variant'), pool_type='tok', patch_size=(16, 16))

  c.teachers = ['prof']  # You could even add multiple.
  c.prof_name = 'vit'
  c.prof_init = init
  c.prof = dict(variant=c.get_ref('variant'), pool_type='tok', patch_size=(16, 16))

  pp_label = '|onehot(1000, key="{lbl}", key_result="labels")|keep("image", "prof", "labels")'
  c.input.pp = (
      f'decode|inception_crop|flip_lr'
      '|copy("image", "prof")'
      f'|resize({c.res})|value_range'
      '|resize(384, key="prof")|value_range(key="prof")'
      + pp_label.format(lbl='label')
  )
  pp_eval_both = (
      'decode|copy("image", "prof")|'
      f'|resize({c.res//7*8})|central_crop({c.res})|value_range'
      f'|resize({384//7*8}, key="prof")|central_crop(384, key="prof")|value_range(key="prof")|'
  )
  pp_eval_student = (
      f'decode|resize({c.res//7*8})|central_crop({c.res})|value_range(-1, 1)'
  )
  pp_eval_prof = (
      f'decode|resize({384//7*8})|central_crop(384)|value_range(outkey="prof")'
  )

  c.mixup = dict(p=1.0, n=2)

  # Distillation settings
  c.distance = 'kl'
  c.distance_kw = dict(t=1.0)

  # Optimizer section
  c.grad_clip_norm = 1.0
  c.optax_name = 'scale_by_adam'
  c.optax = dict(mu_dtype='bfloat16')

  c.lr = 1e-4
  c.wd = 1e-5
  c.schedule = dict(warmup_steps=5000, decay_type='cosine')

  # Define the model parameters which are flexible:
  c.flexi = dict()
  c.flexi.seqhw = dict(
      # The settings to sample from. Corresponding patch-sizes at 240px:
      # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
      v=(5, 6, 8, 10, 12, 15, 16, 20, 24, 30),
      # The probabilities/weights of them. Default uniform.
      p=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
  )

  # Eval section
  def mksplit(split):
    if c.runlocal:
      return split.split('[')[0] + '[:16]'
    return split

  minitrain_split = mksplit('train[:2%]')
  minival_split = mksplit('train[99%:]')
  val_split = mksplit('validation')
  test_split = mksplit('test')
  c.aggressive_cache = False

  def get_eval(s, split, dataset='imagenet2012'):
    return dict(
        type='classification',
        pred=f'student_seqhw={s}',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval_student + pp_label.format(lbl='label'),
        loss_name='sigmoid_xent',
        log_percent=0.05,
        cache_final=False,
    )

  c.evals = {}
  for s in c.flexi.seqhw.v:
    c.evals[f'student_minitrain_{s:02d}'] = get_eval(s, minitrain_split)
    c.evals[f'student_minival_{s:02d}'] = get_eval(s, minival_split)
    c.evals[f'student_val_{s:02d}'] = get_eval(s, val_split)
    c.evals[f'student_v2_{s:02d}'] = get_eval(s, test_split, 'imagenet_v2')
    c.evals[f'student_a_{s:02d}'] = get_eval(s, test_split, 'imagenet_a')
    c.evals[f'student_r_{s:02d}'] = get_eval(s, test_split, 'imagenet_r')
    c.evals[f'student_real_{s:02d}'] = get_eval(s, val_split, 'imagenet2012_real')
    c.evals[f'student_real_{s:02d}'].pp_fn = pp_eval_student + pp_label.format(lbl='real_label')

  def get_eval_t(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        pred='prof',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval_prof + pp_label.format(lbl='label'),
        loss_name='sigmoid_xent',
        log_percent=0.5,  # Teacher is fixed, so eval just for plots.
        cache_final=False,
    )
  c.evals.teacher_minitrain = get_eval_t(minitrain_split)
  c.evals.teacher_minival = get_eval_t(minival_split)
  c.evals.teacher_val = get_eval_t(val_split)
  c.evals.teacher_v2 = get_eval_t(test_split, 'imagenet_v2')
  c.evals.teacher_a = get_eval_t(test_split, 'imagenet_a')
  c.evals.teacher_r = get_eval_t(test_split, 'imagenet_r')
  c.evals.teacher_real = get_eval_t(val_split, 'imagenet2012_real')
  c.evals.teacher_real.pp_fn = pp_eval_prof + pp_label.format(lbl='real_label')

  # Distance evaluators
  def get_dist(split, s):
    return dict(
        type='proj.distill.distance',
        pred=f'student_seqhw={s}_prof',
        data=dict(name='imagenet2012', split=split),
        pp_fn=pp_eval_both + '|keep("image", "prof")',
        log_percent=0.05,
        distances=({'kind': 'kl'}, {'kind': 'logsoftmax_euclidean'},
                   {'kind': 'agree', 'k': 1}, {'kind': 'agree', 'k': 5}),
        cache_final=False,
    )
  for s in c.flexi.seqhw.v:
    c.evals[f'dist_minitrain_{s:02d}'] = get_dist(minitrain_split, s)
    c.evals[f'dist_val_{s:02d}'] = get_dist(val_split, s)

  return c