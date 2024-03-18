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
r"""Pre-training BiT on ILSVRC-2012 as in https://arxiv.org/abs/1912.11370

Run training of a BiT-ResNet-50x1 variant, which takes ~32min on v3-128:

big_vision.train \
    --config big_vision/configs/bit_i1k.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
    --config.model.depth 50 --config.model.width 1
"""

# from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc


def get_config(runlocal=False):
  """Config for training on ImageNet-1k."""
  config = mlc.ConfigDict()

  config.seed = 0
  config.total_epochs = 90
  config.num_classes = 1000
  config.loss = 'softmax_xent'

  config.input = dict()
  config.input.data = dict(
      name='imagenet2012',
      split='train[:99%]',
  )
  config.input.batch_size = 4096 if not runlocal else 32
  config.input.cache_raw = not runlocal  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 250_000 if not runlocal else 10_000  # Per host.

  pp_common = '|onehot(1000, key="{lbl}", key_result="labels")'
  pp_common += '|value_range(-1, 1)|keep("image", "labels")'
  config.input.pp = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common.format(lbl='label')
  pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Model section
  config.model_name = 'bit'
  config.model = dict(
      depth=50,  # You can also pass e.g. [3, 5, 10, 2]
      width=1.0,
  )

  # Optimizer section
  config.optax_name = 'big_vision.momentum_hp'
  config.grad_clip_norm = 1.0

  # linear scaling rule. Don't forget to sweep if sweeping batch_size.
  config.wd = (1e-4 / 256) * config.input.batch_size
  config.lr = (0.1 / 256) * config.input.batch_size
  config.schedule = dict(decay_type='cosine', warmup_steps=1000)

  # Eval section
  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_steps=1000,  # Very fast O(seconds) so it's fine to run it often.
        cache_final=not runlocal,
    )
  config.evals = {}
  config.evals.train = get_eval('train[:2%]')
  config.evals.minival = get_eval('train[99%:]')
  config.evals.val = get_eval('validation')
  config.evals.v2 = get_eval('test', dataset='imagenet_v2')
  config.evals.real = get_eval('validation', dataset='imagenet2012_real')
  config.evals.real.pp_fn = pp_eval.format(lbl='real_label')

  # config.evals.fewshot = get_fewshot_lsr(runlocal=runlocal)
  # config.evals.fewshot.log_steps = 1000

  return config