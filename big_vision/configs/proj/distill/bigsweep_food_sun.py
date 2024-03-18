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
r"""Distilling BiT-R152x2 into BiT-R50x1 on Food101/Sun397 as in https://arxiv.org/abs/2106.05237

While many epochs are required, this is a small dataset, and thus overall it
is still fast and possible to run on the relatively small v3-8TPUs (or GPUs).

This configuration contains the recommended settings from Fig3/Tab4 of the
paper, which can be selected via the fast/medium/long config argument.
(best settings were selected on a 10% minival)

For Food101:
- The `fast` variant takes ~45min on a v2-8 TPU.
  Example logs at gs://big_vision/distill/bit_food_fast_06-19_0547/big_vision_metrics.txt
  Example logs at gs://big_vision/distill/bit_sun_fast_06-20_1839/big_vision_metrics.txt
- The `long` variant takes ~14h on a v3-8 TPU.
  Example logs at gs://big_vision/distill/bit_food_long_06-19_0614/big_vision_metrics.txt
  Example logs at gs://big_vision/distill/bit_sun_long_06-20_1912/big_vision_metrics.txt

big_vision.trainers.proj.distill.distill \
    --config big_vision/configs/proj/distill/bigsweep_food_sun.py:data=food,variant=fast \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
"""

import big_vision.configs.common as bvcc
import big_vision.configs.proj.distill.common as cd
import ml_collections as mlc

H, L = 160, 128
NCLS = dict(food=101, sun=397)


def get_config(arg=None):
  """Config for massive hypothesis-test on pet."""
  arg = bvcc.parse_arg(arg, runlocal=False, data='food', variant='medium', crop='inception_crop(128)')
  config = mlc.ConfigDict()

  config.input = {}
  config.input.data = dict(
      name=dict(food='food101', sun='sun397')[arg.data],
      split=dict(food='train[:90%]', sun='train')[arg.data],
  )
  config.input.batch_size = 512
  config.input.cache_raw = True
  config.input.shuffle_buffer_size = 50_000
  config.prefetch_to_device = 4

  config.num_classes = NCLS[arg.data]
  config.total_epochs = {'fast': 100, 'medium': 1000, 'long': 3000}[arg.variant]

  config.log_training_steps = 50
  config.ckpt_steps = 2500

  # Model section
  config.student_name = 'bit_paper'
  config.student = dict(depth=50, width=1)

  config.teachers = ['prof_m']
  config.prof_m_name = 'bit_paper'
  config.prof_m_init = cd.inits[f'BiT-M R152x2 {arg.data} rc128']
  config.prof_m = dict(depth=152, width=2)

  # Preprocessing pipeline for student & tacher.
  pp_common = (
      '|value_range(-1, 1)'
      f'|onehot({config.num_classes}, key="label", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.input.pp = f'decode|{arg.crop}|flip_lr' + pp_common
  ppv = 'decode|resize_small(160)|central_crop(128)' + pp_common

  config.mixup = dict(p=1.0)

  # Distillation settings
  config.distance = 'kl'
  config.distance_kw = dict(t={
      'food': {'fast': 10., 'medium': 10., 'long': 5.},
      'sun': {'fast': 10., 'medium': 10., 'long': 10.},
  }[arg.data][arg.variant])

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  config.lr = {
      'food': {'fast': 0.01, 'medium': 0.001, 'long': 0.01},
      'sun': {'fast': 0.01, 'medium': 0.001, 'long': 0.01},
  }[arg.data][arg.variant]
  config.wd = {
      'food': {'fast': 1e-3, 'medium': 3e-4, 'long': 1e-4},
      'sun': {'fast': 1e-3, 'medium': 1e-4, 'long': 3e-5},
  }[arg.data][arg.variant]
  config.schedule = dict(warmup_steps=1500, decay_type='cosine')
  config.optim_name = 'adam_hp'

  # Eval section
  minitrain_split = 'train[:1024]' if not arg.runlocal else 'train[:16]'
  if arg.data == 'food':
    val_split = 'train[90%:]' if not arg.runlocal else 'train[:16]'
    test_split = 'validation' if not arg.runlocal else 'test[:16]'
  elif arg.data == 'sun':
    val_split = 'validation' if not arg.runlocal else 'validation[:16]'
    test_split = 'test' if not arg.runlocal else 'test[:16]'

  def get_eval(split):
    return dict(
        type='classification',
        pred='student_fwd',
        data=dict(name=config.input.data.name, split=split),
        pp_fn=ppv,
        loss_name='softmax_xent',
        log_steps=500,
    )
  config.evals = {}
  config.evals.student_train = get_eval(minitrain_split)
  config.evals.student_val = get_eval(val_split)
  config.evals.student_test = get_eval(test_split)

  # Teacher is fixed, so rare evals.
  teacher = dict(log_steps=100_000, pred='prof_m_fwd')
  config.evals.teacher_train = {**config.evals.student_train, **teacher}
  config.evals.teacher_val = {**config.evals.student_val, **teacher}
  config.evals.teacher_test = {**config.evals.student_test, **teacher}

  # Could in principle also look at agreement on other datasets!
  def get_dist(split):
    return dict(
        type='proj.distill.distance',
        pred='student_prof_m_fwd',
        data=dict(name=config.input.data.name, split=split),
        pp_fn=ppv + '|keep("image")',
        log_steps=1000,
        distances=({'kind': 'kl'}, {'kind': 'euclidean'},
                   {'kind': 'agree', 'k': 1}, {'kind': 'agree', 'k': 5}),
    )
  config.evals.dist_train = get_dist(minitrain_split)
  config.evals.dist_val = get_dist(val_split)
  config.evals.dist_test = get_dist(test_split)

  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8

  return config


def get_hyper(hyper):
  """Hyper sweep."""
  # TODO: update, similar to flowers_pet sweep.
  # By default, not running the MASSIVE sweep, just the recommended setting
  # across durations. However, code for sweep is left for reference/convenience.
  return hyper.zipit([
      hyper.sweep('config.total_epochs', [100, 1_000]),
      hyper.sweep('config.mixup.p', [0.0, 1.0]),
      hyper.sweep('config.weight_decay', [1e-3, 1e-5]),
  ])

  # pylint: disable=unreachable

  def fix(**kw):
    return hyper.product([hyper.fixed(f'config.{k}', v, length=1)
                          for k, v in kw.items()])

  def setting(p, l, m, crop, pp_end=None, **extra):
    pp_end = pp_end or (
        f'|value_range(-1, 1, key="image")'
        f'|onehot({NCLS}, key="label", key_result="labels")'
        f'|keep("image", "labels")'
    )
    return hyper.product([
        fix(**{'mixup.p': p}),
        fix(l=l, m=m, crop=crop),
        fix(pp_train=f'decode|{crop}|flip_lr|randaug({l},{m})' + pp_end),
        fix(**extra)
    ])

  # Mixup, Layers and Mag in randaug.
  plm = [(0.0, 0, 0), (0.1, 0, 0), (0.5, 0, 0), (1.0, 0, 0)]
  return hyper.product([
      hyper.sweep('config.total_epochs', [100, 1000, 3000]),
      hyper.sweep('config.lr.base', [0.001, 0.003, 0.01]),
      hyper.sweep('config.distance_kw.t', [1.0, 2.0, 5.0, 10.0]),
      hyper.sweep('config.weight_decay', [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
      hyper.chainit(
          [setting(p=p, l=l, m=m,
                   crop=(f'resize({H})'
                         f'|inception_crop({L}, outkey="student")'
                         f'|central_crop({L}, outkey="teacher")'),
                   pp_end=(
                       f'|value_range(-1, 1, key="student")'
                       f'|value_range(-1, 1, key="teacher")'
                       f'|onehot({NCLS}, key="label", key_result="labels")'
                       f'|keep("student", "teacher", "labels")'))
           for p, l, m in plm] +
          [setting(p=p, l=l, m=m, crop=f'inception_crop({L})') for
           p, l, m in plm],
      )
  ])
