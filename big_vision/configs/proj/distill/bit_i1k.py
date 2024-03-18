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
r"""Distilling BiT-R152x2 into BiT-R50x1 on ILSVRC-2012 as in https://arxiv.org/abs/2106.05237

Note that as per paper title, good results require many epochs and thus
a lot of _patience_. For experimentation/exploration, consider
using the smaller datasets.

300ep take about 15h on a v3-32 TPU, an example log is available at:
  Example logs at gs://big_vision/distill/bit_i1k_300ep_06-16/big_vision_metrics.txt

big_vision.trainers.proj.distill.distill \
    --config big_vision/configs/proj/distill/bit_i1k.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
    --config.total_epochs 1200
"""

import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
import big_vision.configs.proj.distill.common as cd
import ml_collections as mlc


def get_config(arg=None):
  """Config for distilling on ImageNet."""
  arg = bvcc.parse_arg(arg, runlocal=False)
  config = mlc.ConfigDict()

  config.input = {}
  config.input.data = dict(name='imagenet2012', split='train[:98%]')
  config.input.batch_size = 4096
  config.input.shuffle_buffer_size = 250_000

  config.num_classes = 1000
  config.total_epochs = 1200  # A good middle-ground

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 20000

  # Model section
  config.student_name = 'bit_paper'
  config.student = dict(depth=50, width=1)

  config.teachers = ['prof_m']  # You could even add multiple.

  # TODO: use public checkpoint name.
  config.prof_m_name = 'bit_paper'
  config.prof_m_init = cd.inits['BiT-M R152x2 imagenet2012 ic224']
  config.prof_m = dict(depth=152, width=2)

  pp_common = (
      '|value_range(-1, 1)'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.input.pp = (
      'decode_jpeg_and_inception_crop(224)|flip_lr' +
      pp_common.format(lbl='label')
  )
  ppv = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.mixup = dict(p=1.0)

  # Distillation settings
  config.distance = 'kl'
  config.distance_kw = dict(t=1.0)

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  config.lr = 0.03
  config.wd = 0.0003
  config.schedule = dict(warmup_steps=5000, decay_type='cosine')

  # Eval section
  minitrain_split = 'train[:2%]' if not arg.runlocal else 'train[:16]'
  minival_split = 'train[99%:]' if not arg.runlocal else 'train[:16]'
  val_split = 'validation' if not arg.runlocal else 'validation[:16]'
  real_split = 'validation' if not arg.runlocal else 'validation[:16]'
  v2_split = 'test' if not arg.runlocal else 'test[:16]'

  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        pred='student_fwd',
        data=dict(name=dataset, split=split),
        pp_fn=ppv.format(lbl='label'),
        loss_name='softmax_xent',
        log_steps=1000,
    )

  config.evals = {}
  config.evals.student_train = get_eval(minitrain_split)
  config.evals.student_minival = get_eval(minival_split)
  config.evals.student_val = get_eval(val_split)
  config.evals.student_v2 = get_eval(v2_split, dataset='imagenet_v2')
  config.evals.student_real = get_eval(real_split, dataset='imagenet2012_real')
  config.evals.student_real.pp_fn = ppv.format(lbl='real_label')

  config.evals.student_fewshot = get_fewshot_lsr(runlocal=arg.runlocal)
  config.evals.student_fewshot.pred = 'student_fwd'
  config.evals.student_fewshot.log_steps = 10_000

  teacher_eval = dict(
      log_steps=100_000,  # Teacher is fixed, so rare evals.
      pred='prof_m_fwd',
  )
  config.evals.teacher_train = {**config.evals.student_train, **teacher_eval}
  config.evals.teacher_minival = {**config.evals.student_minival, **teacher_eval}
  config.evals.teacher_val = {**config.evals.student_val, **teacher_eval}
  config.evals.teacher_v2 = {**config.evals.student_v2, **teacher_eval}
  config.evals.teacher_real = {**config.evals.student_real, **teacher_eval}
  config.evals.teacher_fewshot = {**config.evals.student_fewshot, **teacher_eval}
  config.evals.teacher_fewshot.prefix = 'z_teacher/'

  # Could in principle also look at agreement on other datasets!
  def get_dist(split, dataset='imagenet2012'):
    return dict(
        type='proj.distill.distance',
        pred='student_prof_m_fwd',
        data=dict(name=dataset, split=split),
        pp_fn=ppv.format(lbl='label') + '|keep("image")',
        log_steps=1000,
        distances=({'kind': 'kl'}, {'kind': 'euclidean'},
                   {'kind': 'agree', 'k': 1}, {'kind': 'agree', 'k': 5}),
    )
  config.evals.dist_train = get_dist(minitrain_split)
  config.evals.dist_minival = get_dist(minival_split)
  config.evals.dist_val = get_dist(val_split)
  config.evals.dist_v2 = get_dist(v2_split, dataset='imagenet_v2')

  # NOTE: CKA evaluator does not work with batch padding, so the size of the
  # split must be a multiple of the batch size.
  def get_cka(split):
    return dict(
        type='proj.distill.cka',
        pred='student_prof_m_fwd',
        data=dict(name='imagenet2012', split=split),
        pp_fn=ppv.format(lbl='label') + '|keep("image")',
        log_steps=1000,
    )
  config.evals.cka_train = get_cka('train[:24576]' if not arg.runlocal else 'train[:16]')
  config.evals.cka_minival = get_cka('train[-24576:]' if not arg.runlocal else 'train[:16]')
  config.evals.cka_val = get_cka('validation[:49152]' if not arg.runlocal else 'validation[:16]')

  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8

  return config