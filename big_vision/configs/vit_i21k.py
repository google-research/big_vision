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
r"""A config for pre-training BiT on ImageNet-21k.

This config relies on the Imagenet-21k tfds dataset, which is not yet
available publicly in TFDS. We intend to add the dataset to public TFDS soon,
and this config will then be runnable.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc

MIXUP_DEF = {
    'none': dict(p=0.0, fold_in=None),
    'light1': dict(p=0.0, fold_in=None),
    'light2': dict(p=0.2, fold_in=None),
    'medium1': dict(p=0.2, fold_in=None),
    'medium2': dict(p=0.5, fold_in=None),
    'strong1': dict(p=0.5, fold_in=None),
    'strong2': dict(p=0.8, fold_in=None),
}

RANDAUG_DEF = {
    'none': '',
    'light1': 'randaug(2,0)',  # Actually not nothing!
    'light2': 'randaug(2,10)',
    'medium1': 'randaug(2,15)',
    'medium2': 'randaug(2,15)',
    'strong1': 'randaug(2,20)',
    'strong2': 'randaug(2,20)',
}


def get_config(arg=None):
  """Config for training."""
  arg = bvcc.parse_arg(arg, variant='B/16', runlocal=False, aug=None)
  config = mlc.ConfigDict()

  # If this gives a KeyError, lookup Fig4 of the paper and add an entry.
  # Note, this here is a good average between 30ep and 300ep, sometimes you coud
  # find a slightly better setting for either of them.
  aug_setting = {
      'Ti/16': 'none',
      'S/32': 'none',
      'S/16': 'light1',
      'B/32': 'light2',
      'B/16': 'light2',
      'L/16': 'medium2',
  }[arg.variant]

  config.dataset = 'imagenet21k'
  config.train_split = 'full[51200:]'
  config.num_classes = 21843
  config.init_head_bias = -10.0
  config.loss = 'sigmoid_xent'

  config.batch_size = 4096
  config.num_epochs = 300

  pp_common = f'|value_range(-1, 1)|onehot({config.num_classes})|keep("image", "labels")'
  config.pp_train = f'decode_jpeg_and_inception_crop(224)|flip_lr|{RANDAUG_DEF[aug_setting]}' + pp_common
  pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  # Aggressive pre-fetching because our models here are small, so we not only
  # can afford it, but we also need it for the smallest models to not be
  # bottle-necked by the input pipeline. Play around with it for -L models tho.
  config.prefetch_to_host = 8
  config.prefetch_to_device = 4

  config.log_training_steps = 50
  config.checkpoint_steps = 1000

  # Model section
  config.model_name = 'vit'
  config.model = dict(variant=arg.variant, pool_type='gap', posemb='learn')

  # Optimizer section
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')
  config.grad_clip_norm = 1.0

  config.lr = 0.001
  config.wd = 0.0001
  config.schedule = dict(warmup_steps=10_000, decay_type='cosine')

  config.mixup = MIXUP_DEF[aug_setting]

  # Eval section
  eval_common = dict(
      dataset=config.dataset,
      pp_fn=pp_eval,
      loss_name=config.loss,
      log_steps=1000,  # Very fast O(seconds) so it's fine to run it often.
  )
  config.evals = {}
  config.evals.test = {**eval_common, 'split': 'full[:25_600]'}
  config.evals.val = {**eval_common, 'split': 'full[25_600:51_200]'}
  config.evals.train = {**eval_common, 'split': 'full[51_200:76_800]'}
  config.evals.fewshot = get_fewshot_lsr(runlocal=arg.runlocal)
  config.evals.fewshot.log_steps = 10_000

  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.shuffle_buffer_size = 10
    config.batch_size = 8
    config.evals.test.split = 'full[:16]'
    config.evals.train.split = 'full[:16]'
    config.evals.val.split = 'full[:16]'

  return config