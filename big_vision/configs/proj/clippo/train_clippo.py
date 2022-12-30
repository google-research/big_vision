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
r"""Trains CLIP with Pixels Only (CLIPPO), https://arxiv.org/abs/2212.08045

IMPORTANT NOTE: This config uses coco_captions by default for demonstration
purposes since the TFDS catalog does not provide any large image/alt-text data
set; the training will not produce a model with useful accuracy. Please
replace the data set below (marked by a comment) with an appropriate image/
alt-text data set wrapped in TFDS (for example LAION-400M) and run the config
with the suffix `:test_with_coco=False` to train on your data set. Refer to
the following guide to build a TFDS wrapper for your favorite image/alt-text
data set:
https://www.tensorflow.org/datasets/add_dataset

Also note that evaluation on ImageNet requires manual TFDS setup, see
https://github.com/google-research/big_vision#preparing-tfds-data


Example training:

big_vision.trainers.proj.image_text.contrastive \
    --config big_vision/configs/proj/clippo/train_clippo.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%Y-%m-%d_%H%M'`

"""

import big_vision.configs.common as bvcc
from big_vision.configs.common_fewshot import get_fewshot_lsr
from big_vision.configs.proj.image_text import common
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, res=224, runlocal=False, variant='B/16',
      test_with_coco=True, i1k_eval=False)
  config = ConfigDict()

  config.input = {}
  if arg.test_with_coco:
    # Use COCO Captions for sanity-checking
    config.input.data = dict(name='coco_captions', split='train')
    val_data = dict(config.input.data)
    val_data['split'] = 'val'
    config.input.batch_size = 4000 if not arg.runlocal else 32
    config.input.shuffle_buffer_size = 50_000  if not arg.runlocal else 50
    config.total_steps = 400 if not arg.runlocal else 10
  else:
    # Please add your favorite image/alt-text dataset here
    config.input.data = None
    val_data = None
    assert config.input.data is not None and val_data is not None, (
        config.input.data, val_data)

    # The value in the paper is 10 * 1024, which requires 128 TPUv3 cores or a
    # memory optimized ViT implementation when running on 128 TPUv2 cores.
    config.input.batch_size = 8 * 1024 if not arg.runlocal else 32
    config.input.shuffle_buffer_size = 250_000  if not arg.runlocal else 50
    config.total_steps = 100_000 if not arg.runlocal else 10

  def tokenizer(inkey, outkey='labels'):
    return (f'render_unifont('
            f'inkey="{inkey}", '
            f'outkey="{outkey}", '
            f'image_size={arg.res}, '
            f'lower=True, '
            f'font_size=16, '
            f'text_brightness=0, '
            f'background_brightness=127)|'
            f'value_range(-1, 1, inkey="{outkey}", outkey="{outkey}")')

  pp_image = f'decode|resize({arg.res})|value_range(-1,1)'
  if arg.test_with_coco:
    # Train with augmentation when sanity-checking
    pp_image_aug = (
        f'decode|resize({arg.res})|flip_lr|randaug(2,10)|value_range(-1,1)')
    config.input.pp = pp_eval = (
        f'{pp_image_aug}|flatten|{tokenizer("captions/text")}|'
        f'keep("image", "labels")')
  else:
    config.input.pp = pp_eval = (
        f'{pp_image}|flatten|{tokenizer("text")}|keep("image", "labels")')

  config.pp_modules = [
      'ops_general', 'ops_image', 'ops_text', 'proj.clippo.pp_ops']

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 5000

  config.loss_use_global_batch = True

  # Define the model
  config.model_name = 'proj.clippo.one_tower'

  config.model = ConfigDict()
  config.model.image_model = 'vit'
  config.model.image = ConfigDict({
      'variant': arg.variant,
      'pool_type': 'map',
      'head_zeroinit': False,
  })

  if arg.test_with_coco:
    # Initialize with ImageNet21k pretrained checkpoint for sanity-checking
    assert arg.variant == 'B/16', arg.variant
    config.model_init = {'image': 'howto-i21k-B/16'}
    config.model_load = {}
    config.model_load['img_load_kw'] = {
        'dont_load': ['^head/.*', '^MAPHead_0/.*', 'cls']}

  config.model.temperature_init = 10.0
  config.model.out_dim = 768

  # Define the optimizer
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.grad_clip_norm = 1.0

  if arg.test_with_coco:
    # Short schedule for sanity-checking
    config.lr = 0.0001
    config.wd = 0.0003
    config.schedule = dict(decay_type='rsqrt',
                           timescale=100,
                           warmup_steps=100 if not arg.runlocal else 5,
                           cooldown_steps=100 if not arg.runlocal else 5)
  else:
    config.lr = 0.001
    config.wd = 0.0001
    config.schedule = dict(decay_type='rsqrt',
                           timescale=10_000,
                           warmup_steps=10_000 if not arg.runlocal else 5,
                           cooldown_steps=10_000 if not arg.runlocal else 5)

  # Eval section (Both few-shot and zero-shot)
  eval_common = dict(
      type='proj.image_text.contrastive',
      use_global_batch=config.loss_use_global_batch,
      log_steps=1000 if not arg.runlocal else 5,
  )
  config.evals = {}
  sub = '[:4]' if arg.runlocal else ''
  config.evals.val = {
      **eval_common,
      'data': val_data,
      'pp_fn': pp_eval,
  }
  config.evals.coco = {
      **eval_common,
      'data': dict(name='coco_captions', split=f'val{sub}'),
      'pp_fn': (
          f'{pp_image}|flatten|{tokenizer("captions/text")}|'
          f'keep("image", "labels")'),
  }

  if arg.i1k_eval:
    # Requires manual download, see
    # https://github.com/google-research/big_vision#preparing-tfds-data
    config.evals.imagenet = {
        **eval_common,
        'data': dict(name='imagenet2012', split=f'validation{sub}'),
        'pp_fn': (
            f'{pp_image}|clip_i1k_label_names|'
            f'{tokenizer("labels")}|keep("image", "labels")'),
    }
    config.evals.disclf = dict(
        type='proj.image_text.discriminative_classifier',
        pp_txt=tokenizer('texts', 'labels'),
        prefix='z/0shot/',
        log_steps=5_000 if not arg.runlocal else 5)

  config.evals.retrieval_coco = common.get_coco(
      pp_img=f'resize({arg.res})|value_range(-1, 1)',
      pp_txt=tokenizer('texts'),
      log_steps=5_000 if not arg.runlocal else 5,
  )

  # Few-shot  metrics
  config.evals.fewshot = get_fewshot_lsr()
  config.evals.fewshot.log_steps = 5_000 if not arg.runlocal else 5
  config.evals.fewshot.representation_layer = 'img/pre_logits'

  config.seed = 0

  return config
