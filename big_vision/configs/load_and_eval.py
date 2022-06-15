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

# pytype: disable=not-writable,attribute-error
# pylint: disable=line-too-long,missing-function-docstring
r"""A config to load and evaluate models.

The runtime varies widely depending on the model, but each one should reproduce
the corresponding paper's numbers.
This configuration makes use of the "arg" to get_config to select which model
to run, so a few examples are given below:

Run and evaluate a BiT-M ResNet-50x1 model that was transferred to i1k:

big_vision.tools.eval_only \
    --config big_vision/configs/load_and_eval.py:name=bit_paper,batch_size=8 \
    --config.model_init M-imagenet2012 --config.model.width 1 --config.model.depth 50

Run and evaluate the recommended ViT-B/32 from "how to train your vit" paper:

big_vision.tools.eval_only \
    --config big_vision/configs/load_and_eval.py:name=vit_i21k,batch_size=8 \
    --config.model.variant B/32 --config.model_init howto-i21k-B/32
"""

import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc


def get_config(arg='name=bit_paper,batch_size=2'):
  arg = bvcc.parse_arg(arg, name='', batch_size=2)
  config = mlc.ConfigDict()
  config.batch_size_eval = arg.batch_size

  # Just calls the function with the name given as `config`.
  # Could also be a giant if-block if you're into that kind of thing.
  globals()[arg.name](config)
  return config


def bit_paper(config):
  # We could omit init_{shapes,types} if we wanted, as they are the default.
  config.init_shapes = [(1, 224, 224, 3)]
  config.init_types = ['float32']
  config.num_classes = 1000

  config.model_name = 'bit_paper'
  config.model_init = 'M-imagenet2012'  # M = i21k, -imagenet2012 = fine-tuned
  config.model = dict(width=1, depth=50)

  config.evals = {}
  config.evals.fewshot = get_fewshot_lsr()

  pp_clf = (
      'decode|resize(384)|value_range(-1, 1)'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  common_clf = dict(
      type='classification',
      loss_name='softmax_xent',
      cache_final=False,  # Only run once, on low-mem machine.
      dataset='imagenet2012_real',
      split='validation',
  )
  config.evals.test = {
      **common_clf,
      'pp_fn': pp_clf.format(lbl='original_label'),
  }
  config.evals.real = {
      **common_clf,
      'pp_fn': pp_clf.format(lbl='real_label'),
  }
  config.evals.v2 = {
      **common_clf,
      'dataset': 'imagenet_v2',
      'split': 'test',
      'pp_fn': pp_clf.format(lbl='label'),
  }


def vit_i1k(config):
  # We could omit init_{shapes,types} if we wanted, as they are the default.
  config.init_shapes = [(1, 224, 224, 3)]
  config.init_types = ['float32']
  config.num_classes = 1000

  config.model_name = 'vit'
  config.model_init = ''  # Will be set in sweep.
  config.model = dict(variant='S/16', pool_type='gap', posemb='sincos2d',
                      rep_size=True)

  config.evals = {}
  config.evals.fewshot = get_fewshot_lsr()
  config.evals.val = dict(
      type='classification',
      dataset='imagenet2012',
      split='validation',
      pp_fn='decode|resize_small(256)|central_crop(224)|value_range(-1, 1)|onehot(1000, key="label", key_result="labels")|keep("image", "labels")',
      loss_name='softmax_xent',
      cache_final=False,  # Only run once, on low-mem machine.
  )


def vit_i21k(config):
  # We could omit init_{shapes,types} if we wanted, as they are the default.
  config.init_shapes = [(1, 224, 224, 3)]
  config.init_types = ['float32']
  config.num_classes = 21843

  config.model_name = 'vit'
  config.model_init = ''  # Will be set in sweep.
  config.model = dict(variant='B/32', pool_type='tok')

  config.evals = {}
  config.evals.fewshot = get_fewshot_lsr()
  config.evals.val = dict(
      type='classification',
      dataset='imagenet21k',
      split='full[:51200]',
      pp_fn='decode|resize_small(256)|central_crop(224)|value_range(-1, 1)|onehot(21843)|keep("image", "labels")',
      loss_name='sigmoid_xent',
      cache_final=False,  # Only run once, on low-mem machine.
  )
