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
r"""Pre-train ViT-g (1B params) on JFT-3B as in https://arxiv.org/abs/2106.04560

To train ViT-G (2B params), simply update the following single line:
  `config.model.variant = 'G/14'`

The code is released for reference purposes.
One can test the code using public ImageNet-1k or ImageNet-21k dataset.

big_vision.train \
    --config big_vision/configs/proj/scaling_laws/train_vit_g.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'`

"""
from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc


def get_config():
  """Rocket config."""
  config = mlc.ConfigDict()

  config.dataset = 'jft_3b'
  config.val_split = 'val'
  config.train_split = 'train'
  config.num_classes = 29_593
  config.init_head_bias = -10.0

  # Fits 32 images per TPUv3 core with ViT-g/14.
  config.batch_size = 4096*4

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes})'
  pp_common += '|keep("image", "labels")'
  config.pp_train = 'inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'resize_small(256)|central_crop(224)' + pp_common
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 50
  config.log_eval_steps = 1000
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.

  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 10_000

  config.prefetch_to_device = 1
  config.trial = 0

  # Model section
  config.model_name = 'vit'
  config.model = mlc.ConfigDict()
  config.model.variant = 'g/14'
  config.model.pool_type = 'map'

  # Optimizer section
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.grad_clip_norm = 1.0
  config.lr = 8e-4
  config.wd = 0.03 * 8e-4
  config.wd_mults = [
      ('.*head/kernel', 100.0),
      ('.*/kernel', 1.0),
  ]
  config.schedule = dict(
      decay_type='rsqrt', timescale=10_000, warmup_steps=10_000,
      cooldown_steps=50_000)
  config.total_steps = 1_000_000

  # Few-shot eval section
  config.evals = {}
  config.evals.fewshot = dict(log_steps=10_000, **get_fewshot_lsr())

  return config
