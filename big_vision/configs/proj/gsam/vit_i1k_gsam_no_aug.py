# Copyright 2022 Big Vision Authors.
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
r"""Pre-training ViT on ILSVRC-2012 with GSAM in https://arxiv.org/abs/2203.08065

Run training of a B/32 model:

big_vision.trainers.proj.gsam.train \
    --config big_vision/configs/proj/gsam/vit_i1k_gsam_no_aug.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'`

"""

import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc

def get_config(arg=None):
  """Config for training."""
  arg = bvcc.parse_arg(arg, variant='B/32', runlocal=False)
  config = mlc.ConfigDict()

  config.dataset = 'imagenet2012'
  config.train_split = 'train[:99%]'
  config.cache_raw = not arg.runlocal  # Needs up to 120GB of RAM!
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.
  config.num_classes = 1000
  config.loss = 'sigmoid_xent'
  config.batch_size = 4096
  config.num_epochs = 300

  pp_common = (
      '|value_range(-1, 1)'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.pp_train = (
      'decode_jpeg_and_inception_crop(224)|flip_lr|' +
      pp_common.format(lbl='label')
  )
  pp = 'decode|resize_small(256)|central_crop(224)' + pp_common

  # Aggressive pre-fetching because our models here are small, so we not only
  # can afford it, but we also need it for the smallest models to not be
  # bottle-necked by the input pipeline. Play around with it for -L models tho.
  config.prefetch_to_host = 8
  config.prefetch_to_device = 4

  config.log_training_steps = 50
  config.checkpoint_steps = 1000

  # Model section
  config.model_name = 'vit'
  config.model = dict(
      variant=arg.variant,
      rep_size=False,
      pool_type='gap',
  )
  config.init_head_bias = -10.0

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='float32')
  # The modified AdaFactor we introduced in https://arxiv.org/abs/2106.04560
  # almost always behaves exactly like adam, but at a fraction of the memory
  # cost (specifically, adam_bf16 = +1.5M, adafactor = +0.5M), hence it is a
  # good idea to try it when you are memory-bound!
  # config.optax_name = 'big_vision.scale_by_adafactor'
  # A good flag to play with when hitting instabilities, is the following:
  # config.optax = dict(beta2_cap=0.95)

  config.lr = 0.003
  config.wd = 0.001 # default is 0.0001; paper used 0.3, effective wd=0.3*lr
  config.schedule = dict(
      warmup_steps=10_000,
      decay_type='linear',
      linear_end=0.01,
  )

  # GSAM settings.
  # Note: when rho_max=rho_min and alpha=0, GSAM reduces to SAM.
  config.gsam = dict(
      rho_max=0.6,
      rho_min=0.1,
      alpha=0.6,
      lr_max=config.get_ref('lr'),
      lr_min=config.schedule.get_ref('linear_end') * config.get_ref('lr'),
  )

  # Eval section
  eval_common = dict(
      type='classification',
      dataset='imagenet2012',
      pp_fn=pp.format(lbl='label'),
      loss_name=config.loss,
      log_steps=2500,  # Very fast O(seconds) so it's fine to run it often.
  )
  config.evals = {}
  config.evals.train = {**eval_common, 'split': 'train[:2%]'}
  config.evals.minival = {**eval_common, 'split': 'train[99%:]'}
  config.evals.val = {**eval_common, 'split': 'validation'}
  config.evals.v2 = {**eval_common, 'dataset': 'imagenet_v2', 'split': 'test'}

  config.evals.real = {**eval_common}
  config.evals.real.dataset = 'imagenet2012_real'
  config.evals.real.split = 'validation'
  config.evals.real.pp_fn = pp.format(lbl='real_label')

  config.fewshot = get_fewshot_lsr(runlocal=arg.runlocal)
  config.fewshot.log_steps = 10_000

  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.shuffle_buffer_size = 10
    config.batch_size = 8
    config.minival.split = 'train[:16]'
    config.val.split = 'validation[:16]'
    config.real.split = 'validation[:16]'
    config.v2.split = 'test[:16]'

  return config
