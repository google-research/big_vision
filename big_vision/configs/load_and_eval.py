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

# pytype: disable=not-writable,attribute-error
# pylint: disable=line-too-long,missing-function-docstring
r"""A config to load and eval key model using the core train.py.

The runtime varies widely depending on the model, but each one should reproduce
the corresponding paper's numbers.
This configuration makes use of the "arg" to get_config to select which model
to run, so a few examples are given below:

Run and evaluate a BiT-M ResNet-50x1 model that was transferred to i1k:

big_vision.train \
    --config big_vision/configs/load_and_eval.py:name=bit_paper,batch_size=8 \
    --config.model_init M-imagenet2012 --config.model.width 1 --config.model.depth 50

Run and evaluate the recommended ViT-B/32 from "how to train your vit" paper:

big_vision.train \
    --config big_vision/configs/load_and_eval.py:name=vit_i21k,batch_size=8 \
    --config.model.variant B/32 --config.model_init howto-i21k-B/32
"""

import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr


def eval_only(config, batch_size, spec_for_init):
  """Set a few configs that turn trainer into (almost) eval-only."""
  config.total_steps = 0
  config.input = {}
  config.input.batch_size = batch_size
  config.input.data = dict(name='bv:dummy', spec=spec_for_init)
  config.optax_name = 'identity'
  config.lr = 0.0

  config.mesh = [('data', -1)]
  config.sharding_strategy = [('params/.*', 'fsdp(axis="data")')]
  config.sharding_rules = [('act_batch', ('data',))]

  return config


def get_config(arg=''):
  config = bvcc.parse_arg(arg, name='bit_paper', batch_size=4)

  # Make the config eval-only by setting some dummies.
  eval_only(config, config.batch_size, spec_for_init=dict(
      image=dict(shape=(224, 224, 3), dtype='float32'),
  ))

  config.evals = dict(fewshot=get_fewshot_lsr())

  # Just calls the function with the name given as `config`.
  # Could also be a giant if-block if you're into that kind of thing.
  globals()[config.name](config)
  return config


def bit_paper(config):
  config.num_classes = 1000

  config.model_name = 'bit_paper'
  config.model_init = 'M-imagenet2012'  # M = i21k, -imagenet2012 = fine-tuned
  config.model = dict(width=1, depth=50)

  def get_eval(split, lbl, dataset='imagenet2012_real'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        loss_name='softmax_xent',
        cache_final=False,  # Only run once, on low-mem machine.
        pp_fn=(
            'decode|resize(384)|value_range(-1, 1)'
            f'|onehot(1000, key="{lbl}", key_result="labels")'
            '|keep("image", "labels")'
        ),
    )
  config.evals.test = get_eval('validation', 'original_label')
  config.evals.real = get_eval('validation', 'real_label')
  config.evals.v2 = get_eval('test', 'label', 'imagenet_v2')


def vit_i1k(config):
  config.num_classes = 1000

  config.model_name = 'vit'
  config.model_init = ''  # Will be set in sweep.
  config.model = dict(variant='S/16', pool_type='gap', posemb='sincos2d',
                      rep_size=True)

  config.evals.val = dict(
      type='classification',
      data=dict(name='imagenet2012', split='validation'),
      pp_fn='decode|resize_small(256)|central_crop(224)|value_range(-1, 1)|onehot(1000, key="label", key_result="labels")|keep("image", "labels")',
      loss_name='softmax_xent',
      cache_final=False,  # Only run once, on low-mem machine.
  )


def mlp_mixer_i1k(config):
  config.num_classes = 1000

  config.model_name = 'mlp_mixer'
  config.model_init = ''  # Will be set in sweep.
  config.model = dict(variant='L/16')

  config.evals.val = dict(
      type='classification',
      data=dict(name='imagenet2012', split='validation'),
      pp_fn='decode|resize_small(256)|central_crop(224)|value_range(-1, 1)|onehot(1000, key="label", key_result="labels")|keep("image", "labels")',
      loss_name='softmax_xent',
      cache_final=False,  # Only run once, on low-mem machine.
  )


def vit_i21k(config):
  config.num_classes = 21843

  config.model_name = 'vit'
  config.model_init = ''  # Will be set in sweep.
  config.model = dict(variant='B/32', pool_type='tok')

  config.evals.val = dict(
      type='classification',
      data=dict(name='imagenet21k', split='full[:51200]'),
      pp_fn='decode|resize_small(256)|central_crop(224)|value_range(-1, 1)|onehot(21843)|keep("image", "labels")',
      loss_name='sigmoid_xent',
      cache_final=False,  # Only run once, on low-mem machine.
  )
