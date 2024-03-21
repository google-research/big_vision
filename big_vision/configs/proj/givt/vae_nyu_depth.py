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
r"""Train VAE on NYU depth data for GIVT-based UViM.
"""

import big_vision.configs.common as bvcc
import ml_collections as mlc


QUANTIZATION_BINS = 256
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0


def get_config(arg='res=512,patch_size=16'):
  """Config for training label compression on NYU depth."""
  arg = bvcc.parse_arg(arg, res=512, patch_size=16,
                       runlocal=False, singlehost=False)
  config = mlc.ConfigDict()

  config.input = {}
  config.input.data = dict(name='nyu_depth_v2', split='train')

  config.input.batch_size = 1024
  config.input.shuffle_buffer_size = 25_000

  config.total_epochs = 200

  config.input.pp = (
      f'decode|nyu_depth|'
      f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
      f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'bin_nyu_depth(min_depth={MIN_DEPTH}, max_depth={MAX_DEPTH}, num_bins={QUANTIZATION_BINS})|'
      f'value_range(-1, 1)|copy("labels", "image")|keep("image")'
  )
  pp_eval = (
      f'decode|nyu_depth|nyu_eval_crop|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'bin_nyu_depth(min_depth={MIN_DEPTH}, max_depth={MAX_DEPTH}, num_bins={QUANTIZATION_BINS})|'
      f'value_range(-1, 1)|copy("labels", "image")|keep("image")'
  )
  pp_pred = (
      f'decode|nyu_depth|nyu_eval_crop|copy("labels","ground_truth")|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'bin_nyu_depth(min_depth={MIN_DEPTH}, max_depth={MAX_DEPTH}, num_bins={QUANTIZATION_BINS})|'
      f'value_range(-1, 1)|copy("labels", "image")|'
      f'keep("image", "ground_truth")'
  )

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None

  # Model section
  config.min_depth = MIN_DEPTH
  config.max_depth = MAX_DEPTH
  config.model_name = 'proj.givt.vit'
  config.model = mlc.ConfigDict()
  config.model.input_size = (arg.res, arg.res)
  config.model.patch_size = (arg.patch_size, arg.patch_size)
  config.model.code_len = 256
  config.model.width = 768
  config.model.enc_depth = 6
  config.model.dec_depth = 12
  config.model.mlp_dim = 3072
  config.model.num_heads = 12
  config.model.codeword_dim = 16
  config.model.code_dropout = 'none'
  config.model.bottleneck_resize = True
  config.model.scan = True
  config.model.remat_policy = 'nothing_saveable'
  config.model_init = ''

  config.rec_loss_fn = 'xent'  # xent, l2
  config.mask_zero_target = True
  # values: (index in source image, number of classes)
  config.model.inout_specs = {
      'depth': (0, QUANTIZATION_BINS),
  }

  config.beta = 2e-4
  config.beta_percept = 0.0

  # Optimizer section
  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)

  # FSDP training by default
  config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  config.sharding_rules = [('act_batch', ('data',))]

  config.lr = 1e-3
  config.wd = 1e-4
  config.schedule = dict(decay_type='cosine', warmup_steps=0.1)
  config.grad_clip_norm = 1.0

  # Evaluation section
  config.evals = {}
  config.evals.val = mlc.ConfigDict()
  config.evals.val.type = 'mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = {**config.input.data}
  config.evals.val.data.split = 'validation'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 250

  base = {
      'type': 'proj.givt.nyu_depth',
      'data': {**config.input.data},
      'pp_fn': pp_pred,
      'pred': 'predict_depth',
      'log_steps': 2000,
      'min_depth': MIN_DEPTH,
      'max_depth': MAX_DEPTH,
  }
  config.evals.nyu_depth_val = {**base}
  config.evals.nyu_depth_val.data.split = 'validation'

  # ### Uses a lot of memory
  # config.evals.save_pred = dict(type='proj.givt.save_predictions')
  # config.evals.save_pred.pp_fn = pp_eval
  # config.evals.save_pred.log_steps = 100_000
  # config.evals.save_pred.data = {**config.input.data}
  # config.evals.save_pred.data.split = 'validation[:64]'
  # config.evals.save_pred.batch_size = 64
  # config.evals.save_pred.outfile = 'inference.npz'

  config.eval_only = False
  config.seed = 0

  if arg.singlehost:
    config.input.batch_size = 128
    config.num_epochs = 50
  elif arg.runlocal:
    config.input.batch_size = 16
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.enc_depth = 1
    config.model.dec_depth = 1
    config.evals.val.data.split = 'validation[:16]'
    config.evals.val.log_steps = 20
    config.evals.nyu_depth_val.data.split = 'validation[:16]'

  return config