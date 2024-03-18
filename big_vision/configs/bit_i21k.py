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
r"""A config for pre-training BiT on ImageNet-21k.

This config relies on the Imagenet-21k tfds dataset, which is not yet
available publicly in TFDS. We intend to add the dataset to public TFDS soon,
and this config will then be runnable.
"""

from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc


def get_config():
  """Config for training on imagenet-21k."""
  config = mlc.ConfigDict()

  config.seed = 0
  config.total_epochs = 90
  config.num_classes = 21843
  config.init_head_bias = -10.0
  config.loss = 'sigmoid_xent'

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
  config.input.pp = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common_i21k
  pp_eval = 'decode|resize_small(256)|central_crop(224)'

  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Model section
  config.model_name = 'bit_paper'
  config.model = dict(depth=50, width=1.0)

  # Optimizer section
  config.optax_name = 'big_vision.momentum_hp'
  config.grad_clip_norm = 1.0

  # linear scaling rule. Don't forget to sweep if sweeping batch_size.
  config.lr = (0.03 / 256) * config.input.batch_size
  config.wd = (3e-5 / 256) * config.input.batch_size
  config.schedule = dict(decay_type='cosine', warmup_steps=5000)

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
  config.evals.fewshot = get_fewshot_lsr()
  config.evals.fewshot.log_steps = 25_000

  return config