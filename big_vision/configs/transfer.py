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

# pylint: disable=line-too-long,missing-function-docstring
r"""A config for transferring vit-augreg.

Best HP selected on (mini)val, expected test results (repeated 5 times):

ViT-Augreg-B/32:
    Dataset, crop, learning rate, mean (%), range (%)
  - ImageNet, inception_crop, 0.03, 83.27, [83.22...83.33]
  - Cifar10, resmall_crop, 0.003, 98.55, [98.46...98.6]
  - Cifar100, resmall_crop, 0.01, 91.35, [91.09...91.62]
  - Pets, inception_crop, 0.003, 93.78, [93.62...94.00]
  - Flowers, inception_crop, 0.003, 99.43, [99.42...99.45]


Command to run:
big_vision.train \
    --config big_vision/configs/transfer.py:model=vit-i21k-augreg-b/32,dataset=cifar10,crop=resmall_crop \
    --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03
"""

import big_vision.configs.common as bvcc
import ml_collections as mlc


def _set_model(config, model):
  """Load pre-trained models: vit or bit."""
  # Reset the head to init (of zeros) when transferring.
  config.model_load = dict(dont_load=['head/kernel', 'head/bias'])

  if model == 'vit-i21k-augreg-b/32':
    # Load "recommended" upstream B/32 from https://arxiv.org/abs/2106.10270
    config.model_name = 'vit'
    config.model_init = 'howto-i21k-B/32'
    config.model = dict(variant='B/32', pool_type='tok')
  elif model == 'vit-i21k-augreg-l/16':
    config.model_name = 'vit'
    config.model_init = 'howto-i21k-L/16'
    config.model = dict(variant='L/16', pool_type='tok')
  elif model == 'vit-s16':
    config.model_name = 'vit'
    config.model_init = 'i1k-s16-300ep'
    config.model = dict(variant='S/16', pool_type='gap', posemb='sincos2d',
                        rep_size=True)
  elif model == 'bit-m-r50x1':
    config.model_name = 'bit_paper'
    config.model_init = 'M'
    config.model = dict(depth=50, width=1)
  else:
    raise ValueError(f'Unknown model: {model}, please define customized model.')


def _set_dataset(config, dataset, crop='inception_crop', h_res=448, l_res=384):
  if dataset == 'cifar10':
    _set_task(config, 'cifar10', 'train[:98%]', 'train[98%:]', 'test', 10, steps=10_000, warmup=500, crop=crop, h_res=h_res, l_res=l_res)
  elif dataset == 'cifar100':
    _set_task(config, 'cifar100', 'train[:98%]', 'train[98%:]', 'test', 100, steps=10_000, warmup=500, crop=crop, h_res=h_res, l_res=l_res)
  elif dataset == 'imagenet2012':
    _set_task(config, 'imagenet2012', 'train[:99%]', 'train[99%:]', 'validation', 1000, steps=20_000, warmup=500, crop=crop, h_res=h_res, l_res=l_res)
    _set_imagenet_variants(config)
  elif dataset == 'oxford_iiit_pet':
    _set_task(config, 'oxford_iiit_pet', 'train[:90%]', 'train[90%:]', 'test', 37, steps=500, warmup=100, crop=crop, h_res=h_res, l_res=l_res)
  elif dataset == 'oxford_flowers102':
    _set_task(config, 'oxford_flowers102', 'train[:90%]', 'train[90%:]', 'test', 102, steps=500, warmup=100, crop=crop, h_res=h_res, l_res=l_res)
  else:
    raise ValueError(
        f'Unknown dataset: {dataset}, please define customized dataset.')


def _set_task(config, dataset, train, val, test, n_cls,
              steps=20_000, warmup=500, lbl='label', crop='resmall_crop',
              flip=True, h_res=448, l_res=384):
  """Vision task with val and test splits."""
  config.total_steps = steps
  config.schedule = dict(
      warmup_steps=warmup,
      decay_type='cosine',
  )

  config.input.data = dict(name=dataset, split=train)
  pp_common = (
      '|value_range(-1, 1)|'
      f'onehot({n_cls}, key="{lbl}", key_result="labels")|'
      'keep("image", "labels")'
  )

  if crop == 'inception_crop':
    pp_train = f'decode|inception_crop({l_res})'
  elif crop == 'resmall_crop':
    pp_train = f'decode|resize_small({h_res})|random_crop({l_res})'
  elif crop == 'resize_crop':
    pp_train = f'decode|resize({h_res})|random_crop({l_res})'
  else:
    raise ValueError(f'Unknown crop: {crop}. Must be one of: '
                     'inception_crop, resmall_crop, resize_crop')
  if flip:
    pp_train += '|flip_lr'
  config.input.pp = pp_train + pp_common

  pp = f'decode|resize_small({h_res})|central_crop({l_res})' + pp_common
  config.num_classes = n_cls

  def get_eval(split):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        loss_name='softmax_xent',
        log_steps=100,
        pp_fn=pp,
    )
  config.evals = dict(val=get_eval(val), test=get_eval(test))


def _set_imagenet_variants(config, h_res=448, l_res=384):
  """Evaluation tasks on ImageNet variants: v2 and real."""
  pp = (f'decode|resize_small({h_res})|central_crop({l_res})'
        '|value_range(-1, 1)|onehot(1000, key="{lbl}", key_result="labels")|'
        'keep("image", "labels")'
        )

  # Special-case rename for i1k (val+test -> minival+val)
  config.evals.minival = config.evals.val
  config.evals.val = config.evals.test
  # NOTE: keep test == val for convenience in subsequent analysis.

  config.evals.real = dict(type='classification')
  config.evals.real.data = dict(name='imagenet2012_real', split='validation')
  config.evals.real.pp_fn = pp.format(lbl='real_label')
  config.evals.real.loss_name = config.loss
  config.evals.real.log_steps = 100

  config.evals.v2 = dict(type='classification')
  config.evals.v2.data = dict(name='imagenet_v2', split='test')
  config.evals.v2.pp_fn = pp.format(lbl='label')
  config.evals.v2.loss_name = config.loss
  config.evals.v2.log_steps = 100


def get_config(arg=None):
  """Config for adaptation."""
  arg = bvcc.parse_arg(arg, model='vit', dataset='cifar10', crop='resmall_crop',
                       h_res=448, l_res=384, batch_size=512, fsdp=False,
                       runlocal=False)
  config = mlc.ConfigDict()

  config.input = {}
  config.input.batch_size = arg.batch_size if not arg.runlocal else 8
  config.input.shuffle_buffer_size = 50_000 if not arg.runlocal else 100

  config.log_training_steps = 10
  config.ckpt_steps = 1000
  config.ckpt_timeout = 600

  # Optimizer section
  config.optax_name = 'big_vision.momentum_hp'
  config.grad_clip_norm = 1.0
  config.wd = None  # That's our default, but just being explicit here!
  config.loss = 'softmax_xent'
  config.lr = 0.01
  config.mixup = dict(p=0.0)

  config.seed = 0

  _set_dataset(config, arg.dataset, arg.crop, arg.h_res, arg.l_res)

  _set_model(config, arg.model)
  if arg.fsdp:
    config.mesh = [('data', -1)]
    config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
    config.sharding_rules = [('act_batch', ('data',))]
    config.model.scan = True

  return config