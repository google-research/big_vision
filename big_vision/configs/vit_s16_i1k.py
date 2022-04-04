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
r"""Pre-training ViT-S/16 on ILSVRC-2012 following [arxiv link].

This should take 6-7h to finish 90ep on a TPU-v3-8 and reach 76.5%,
see the tech report for more details.

Command to run:

big_vision.train \
    --config big_vision/configs/ilsvrc_vit_s16.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'`

To run for 300ep, add `--config.num_epochs 300` to the command.
"""

import ml_collections as mlc


def get_config():
  """Config for training."""
  config = mlc.ConfigDict()

  config.dataset = 'imagenet2012'
  config.train_split = 'train[:99%]'
  config.cache_raw = True  # Requires up to 120GB of RAM!
  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.
  config.num_classes = 1000
  config.loss = 'softmax_xent'
  config.batch_size = 1024
  config.num_epochs = 90

  pp_common = (
      '|value_range(-1, 1)'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.pp_train = (
      'decode_jpeg_and_inception_crop(224)|flip_lr|randaug(2,10)' +
      pp_common.format(lbl='label')
  )
  pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.log_training_steps = 50
  config.log_eval_steps = 1000
  config.checkpoint_steps = 1000

  # Model section
  config.model_name = 'vit'
  config.model = dict(
      variant='S/16',
      rep_size=True,
      pool_type='gap',
      posemb='sincos2d',
  )

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  config.lr = 0.001
  config.wd = 0.0001
  config.schedule = dict(warmup_steps=10_000, decay_type='cosine')

  config.mixup = dict(p=0.2, fold_in=None)

  # Eval section
  config.evals = [
      ('minival', 'classification'),
      ('val', 'classification'),
      ('real', 'classification'),
      ('v2', 'classification'),
  ]

  eval_common = dict(
      pp_fn=pp_eval.format(lbl='label'),
      loss_name=config.loss,
      log_steps=1000,
  )

  config.minival = dict(**eval_common)
  config.minival.dataset = 'imagenet2012'
  config.minival.split = 'train[99%:]'

  config.val = dict(**eval_common)
  config.val.dataset = 'imagenet2012'
  config.val.split = 'validation'

  config.real = dict(**eval_common)
  config.real.dataset = 'imagenet2012_real'
  config.real.split = 'validation'
  config.real.pp_fn = pp_eval.format(lbl='real_label')

  config.v2 = dict(**eval_common)
  config.v2.dataset = 'imagenet_v2'
  config.v2.split = 'test'

  return config
