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
r"""A config for training a UViM stage I model for the depth task.
"""

import itertools
import big_vision.configs.common as bvcc
import ml_collections as mlc


QUANTIZATION_BINS = 256
# Depths outside of this range will not be evaluated.
MIN_DEPTH = 1e-3
MAX_DEPTH = 10


def get_config(arg='res=512,patch_size=16'):
  """Config for training label compression on NYU depth v2."""
  arg = bvcc.parse_arg(arg, res=512, patch_size=16,
                       runlocal=False, singlehost=False)
  config = mlc.ConfigDict()

  config.task = 'proj.uvim.depth_task'

  config.input = {}
  config.input.data = dict(name='nyu_depth_v2', split='train)

  config.input.batch_size = 1024
  config.input.shuffle_buffer_size = 25_000

  config.total_epochs = 200

  config.input.pp = (
      f'decode|nyu_depth|'
      f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
      f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'value_range(-1, 1)|keep("image","labels")'
  )

  pp_eval = (
      f'decode|nyu_depth|nyu_eval_crop|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'value_range(-1, 1)|keep("image","labels")'
  )

  # There are no image IDs in TFDS, so hand through the ground truth for eval.
  pp_pred = (
      f'nyu_depth|nyu_eval_crop|copy("labels","ground_truth")|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'value_range(-1, 1)|'
      f'keep("image","labels","ground_truth")'
  )

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 20_000

  # Model section
  config.min_depth = MIN_DEPTH
  config.max_depth = MAX_DEPTH
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
      'depth': (QUANTIZATION_BINS, arg.patch_size**2),
  }
  config.model.outputs = config.model.inputs

  # VQVAE-specific params.
  config.freeze_dict = False  # Will freeze a dict. inside VQ-VAE model.
  config.w_commitment = 0.0

  # Optimizer section
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)

  config.lr = 1e-3
  config.wd = 1e-5
  config.schedule = dict(decay_type='cosine', warmup_steps=4_000)
  config.grad_clip_norm = 1.0

  # Evaluation section
  config.evals = {}
  config.evals.val = mlc.ConfigDict()
  config.evals.val.type = 'proj.uvim.compute_mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = {**config.input.data}
  config.evals.val.data.split = 'validation'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 250

  base = {
      'type': 'proj.uvim.nyu_depth',
      'dataset': config.input.data.name,
      'pp_fn': pp_pred,
      'log_steps': 2000,
      'min_depth': MIN_DEPTH,
      'max_depth': MAX_DEPTH,
  }
  config.evals.nyu_depth_val = dict(**base, split='validation')

  config.seed = 0

  if arg.singlehost:
    config.input.batch_size = 128
    config.total_epochs = 50
  elif arg.runlocal:
    config.input.batch_size = 16
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.enc_depth = 1
    config.model.dec_depth = 1
    config.evals.val.data.split = 'validation[:16]'
    config.evals.val.log_steps = 20

  return config