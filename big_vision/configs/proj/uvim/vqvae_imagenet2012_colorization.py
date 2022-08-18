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
r"""A config for training a colorization VQ-VAE on imagenet2012.
"""

import itertools
import big_vision.configs.common as bvcc
import ml_collections as mlc


def get_config(arg='res=512,patch_size=16'):
  """A config for training a UViM stage I model for the colorization task."""
  arg = bvcc.parse_arg(arg, res=512, patch_size=16,
                       runlocal=False, singlehost=False)
  config = mlc.ConfigDict()

  config.task = 'proj.uvim.colorization_task'

  config.input = {}
  config.input.data = dict(name='imagenet2012', split='train[4096:]')

  config.input.batch_size = 1024
  config.input.shuffle_buffer_size = 25_000

  config.total_epochs = 100

  config.input.pp = (
      f'decode_jpeg_and_inception_crop({arg.res})'
      f'|flip_lr'
      f'|copy(inkey="image", outkey="labels")'
      f'|rgb_to_grayscale_to_rgb(inkey="image",outkey="image")'
      f'|value_range(-1,1,key="image")'
      f'|value_range(-1,1,key="labels")'
      f'|keep("image","labels")')

  pp_eval = (
      f'decode'
      f'|resize({arg.res})'
      f'|copy(inkey="image", outkey="labels")'
      f'|rgb_to_grayscale_to_rgb(inkey="image",outkey="image")'
      f'|value_range(-1,1,key="image")'
      f'|value_range(-1,1,key="labels")'
      f'|keep("image","labels")')

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 20_000

  # Model section
  config.model_name = 'proj.uvim.vit'
  config.model = mlc.ConfigDict()
  config.model.input_size = (arg.res, arg.res)
  config.model.patch_size = (arg.patch_size, arg.patch_size)
  config.model.code_len = 256
  config.model.width = 768
  config.model.enc_depth = 6
  config.model.dec_depth = 12
  config.model.mlp_dim = 3072
  config.model.num_heads = 12
  config.model.dict_size = 4096  # Number of words in dict.
  config.model.codeword_dim = 768
  config.model.dict_momentum = 0.995  # Momentum for dict. learning.
  config.model.with_encoder_ctx = True
  config.model.with_decoder_ctx = True
  config.model.code_dropout = 'random'
  config.model.bottleneck_resize = True
  config.model.inputs = {
      'color': (3, arg.patch_size**2),
  }
  config.model.outputs = config.model.inputs

  # VQVAE-specific params.
  config.freeze_dict = False  # Will freeze a dict. inside VQ-VAE model.
  config.w_commitment = 0.0

  # Optimizer section
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)

  config.lr = 4e-4
  config.wd = 4e-5
  config.schedule = dict(decay_type='cosine', warmup_steps=4_000)
  config.grad_clip_norm = 1.0

  # Evaluation section
  config.evals = {}
  config.evals.val = mlc.ConfigDict()
  config.evals.val.type = 'proj.uvim.compute_mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = {**config.input.data}
  config.evals.val.data.split = 'train[:4096]'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 250

  base = {
      'type': 'proj.uvim.psnr',
      'pp_fn': pp_eval.replace('decode|', ''),
      'log_steps': 10_000,
  }
  config.evals.psnr_train = dict(**base, split='train[4096:8192]')
  config.evals.psnr_holdout = dict(**base, split='train[:4096]')
  config.evals.psnr_val = dict(**base, split='validation')

  config.evals.colorization_val_coltran_fid = {
      'type': 'proj.uvim.coltran_fid',
      'log_steps': 100_000,
  }

  # config.evals.save_pred = dict(type='proj.uvim.save_predictions')
  # config.evals.save_pred.pp = pp_eval.replace('decode|', '')
  # config.evals.save_pred.log_steps = 100_000
  # config.evals.save_pred.dataset = config.dataset
  # config.evals.save_pred.split = 'validation[:1024]'
  # config.evals.save_pred.outfile = 'inference.npz'

  config.seed = 0

  if arg.singlehost:
    config.input.batch_size = 128
    config.total_epochs = 20
  elif arg.runlocal:
    config.input.batch_size = 16
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.enc_depth = 1
    config.model.dec_depth = 1
    config.evals.val.data.split = 'validation[:16]'
    config.evals.val.log_steps = 20
    config.evals.psnr_train.split = 'train[:256]'
    config.evals.psnr_train.log_steps = 20
    config.evals.psnr_holdout.split = 'train[256:512]'
    config.evals.psnr_holdout.log_steps = 20
    config.evals.psnr_val.split = 'train[:256]'
    config.evals.psnr_val.log_steps = 20
    config.evals.colorization_val_coltran_fid.split = 'validation[:256]'
    config.evals.colorization_val_coltran_fid.log_steps = 20

  return config