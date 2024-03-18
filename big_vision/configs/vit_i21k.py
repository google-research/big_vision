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
r"""Pre-training ViT on ImageNet-21k as in https://arxiv.org/abs/2106.10270

This config relies on the Imagenet-21k tfds dataset, which is not yet
available publicly in TFDS. We intend to add the dataset to public TFDS soon,
and this config will then be runnable.

Note that regularization (dropout, stochastic depth) is not currently
implemented. This was not beneficial for ImageNet-21k pre-trainning.
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

  config.seed = 0
  config.total_epochs = 300
  config.num_classes = 21843
  config.init_head_bias = -10.0
  config.loss = 'sigmoid_xent'

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

  config.input = dict()
  config.input.data = dict(
      name='imagenet21k',
      split='full[51200:]',
  )
  config.input.batch_size = 4096
  config.input.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  pp_common = '|value_range(-1, 1)|onehot({onehot_args})|keep("image", "labels")'
  pp_common_i21k = pp_common.format(onehot_args=f'{config.num_classes}')
  pp_common_i1k = pp_common.format(onehot_args='1000, key="label", key_result="labels"')
  config.input.pp = f'decode_jpeg_and_inception_crop(224)|flip_lr|{RANDAUG_DEF[aug_setting]}' + pp_common_i21k
  pp_eval = 'decode|resize_small(256)|central_crop(224)'

  # To continue using the near-defunct randaug op.
  config.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'archive.randaug']

  # Aggressive pre-fetching because our models here are small, so we not only
  # can afford it, but we also need it for the smallest models to not be
  # bottle-necked by the input pipeline. Play around with it for -L models tho.
  config.input.prefetch = 8
  config.prefetch_to_device = 4

  config.log_training_steps = 50
  config.ckpt_steps = 1000

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

  # Evaluations on i21k itself.
  def eval_i21k(split):
    return dict(
        type='classification',
        data={**config.input.data, 'split': split},
        pp_fn=pp_eval + pp_common_i21k,
        loss_name=config.loss,
        log_steps=1000,  # Very fast O(seconds) so it's fine to run it often.
    )
  config.evals = {}
  config.evals.test = eval_i21k('full[:25_600]')
  config.evals.val = eval_i21k('full[25_600:51_200]')
  config.evals.train = eval_i21k('full[51_200:76_800]')

  # Few-shot evaluators
  config.evals.fewshot = get_fewshot_lsr(runlocal=arg.runlocal)
  config.evals.fewshot.log_steps = 25_000

  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8
    config.evals.test.data.split = 'full[:16]'
    config.evals.train.data.split = 'full[:16]'
    config.evals.val.data.split = 'full[:16]'
    config.evals.i1k_val.data.split = 'validation[:16]'
    config.evals.i1k_v2.data.split = 'test[:16]'
    config.evals.i1k_a.data.split = 'test[:16]'
    config.evals.i1k_r.data.split = 'test[:16]'

  return config