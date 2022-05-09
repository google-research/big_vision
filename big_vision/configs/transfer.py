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
    # Load "recommented" upstream B/32 from https://arxiv.org/abs/2106.10270
    config.model_name = 'vit'
    config.model_init = 'howto-i21k-B/32'
    config.model = dict(variant='B/32', pool_type='tok')
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

  config.dataset = dataset
  config.train_split = train
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
  config.pp_train = pp_train + pp_common

  pp = f'decode|resize_small({h_res})|central_crop({l_res})' + pp_common
  config.num_classes = n_cls
  config.evals = [('val', 'classification'),
                  ('test', 'classification')]

  eval_common = dict(
      dataset=dataset,
      loss_name='softmax_xent',
      log_steps=100,
      pp_fn=pp,
  )
  config.val = dict(**eval_common)
  config.val.split = val
  config.test = dict(**eval_common)
  config.test.split = test


def _set_imagenet_variants(config, h_res=448, l_res=384):
  """Evaluation tasks on ImageNet variants: v2 and real."""
  pp = (f'decode|resize_small({h_res})|central_crop({l_res})'
        '|value_range(-1, 1)|onehot(1000, key="{lbl}", key_result="labels")|'
        'keep("image", "labels")'
        )
  config.evals = [
      ('minival', 'classification'),
      ('val', 'classification'),
      ('real', 'classification'),
      ('v2', 'classification'),
  ]

  # Special-case rename for i1k (val+test -> minival+val)
  config.minival, config.val = config.val, config.test
  del config.test

  config.real = dict()
  config.real.dataset = 'imagenet2012_real'
  config.real.split = 'validation'
  config.real.pp_fn = pp.format(lbl='real_label')
  config.real.loss_name = config.loss
  config.real.log_steps = 100

  config.cls_v2 = dict()
  config.cls_v2.dataset = 'imagenet_v2'
  config.cls_v2.split = 'test'
  config.cls_v2.pp_fn = pp.format(lbl='label')
  config.cls_v2.loss_name = config.loss
  config.cls_v2.log_steps = 100


def get_config(arg=None):
  """Config for adaptation."""
  arg = bvcc.parse_arg(arg, model='vit', dataset='cifar10', crop='resmall_crop',
                       h_res=448, l_res=384, runlocal=False)
  config = mlc.ConfigDict()

  config.batch_size = 512 if not arg.runlocal else 8
  config.shuffle_buffer_size = 50_000 if not arg.runlocal else 100

  config.log_training_steps = 10
  config.log_eval_steps = 100  # It's very fast, frequent is ok.
  config.checkpoint_steps = 1000
  config.checkpoint_timeout = 600

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

  return config