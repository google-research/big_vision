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

# pytype: disable=attribute-error,line-too-long
r"""Jet config for imagenet64.

Expected values in imagenet64 (200 epochs):
  - 32 couplings and block depth 2: 3.72 bpd
  - 64 couplings and block depth 5: 3.66 bpd
"""

import big_vision.configs.common as bvcc


def get_config(arg=None):
  """Config for training a Flow model."""
  config = bvcc.parse_arg(arg, mode='')

  config.seed = 0
  config.total_epochs = 200

  config.input = dict()
  config.input.data = dict(
      name='downsampled_imagenet/64x64',
      split='train',
  )
  config.input.batch_size = 1024
  config.input.shuffle_buffer_size = 250_000

  config.input.pp = 'decode|resize(64)|value_range(-1, 1)|keep("image")'
  pp_eval = 'decode|resize(64)|value_range(-1, 1)|keep("image")'

  config.log_training_steps = 50
  config.ckpt_steps = 5000

  # Model section
  config.model_name = 'proj.jet.jet'
  config.model = dict(
      depth=32, block_depth=2, emb_dim=512, num_heads=8,
      kinds=('channels', 'channels', 'channels', 'channels', 'spatial'),
      channels_coupling_projs=('random',),
      spatial_coupling_projs=('checkerboard', 'checkerboard-inv',
                              'vstripes', 'vstripes-inv',
                              'hstripes', 'hstripes-inv'))

  # Optimizer section
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16', b2=0.95)
  config.grad_clip_norm = 1.0

  # rsqrt schedule.
  config.lr = 3e-4
  config.wd = 1e-5
  config.wd_mults = (
      ('.*', 1.0),
  )
  config.schedule = [
      ('.*FREEZE_ME.*', None),  # Permutation matrices should be always frozen.
      ('.*', dict(decay_type='cosine', warmup_percent=0.1)),
  ]

  # config.mesh = [('replica', 16), ('fsdp', -1)]
  # config.sharding_strategy = [('.*', 'fsdp(axis="fsdp")')]
  # config.sharding_rules = [('act_batch', ('replica', 'fsdp'))]

  # Eval section
  config.evals = {}

  config.evals.minitrain_bits = dict(
      type='mean',
      pred='loss',
      data=dict(name=config.input.data.name, split='train[:4096]'),
      pp_fn=pp_eval,
      log_percent=0.05,
      )

  config.evals.val_bits = dict(
      type='mean',
      pred='loss',
      data=dict(name=config.input.data.name, split='validation'),
      pp_fn=pp_eval,
      log_percent=0.05,
      )

  if config.mode == 'runlocal':
    del config.total_epochs
    config.total_steps = 200
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 32
    config.model.depth = 1
    config.model.block_depth = 1

    config.evals.val_bits.data.split = 'validation[:16]'
    config.evals.minitrain_bits.data.split = 'train[:16]'

  return config
