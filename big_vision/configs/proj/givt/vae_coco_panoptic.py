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
r"""Train VAE for GIVT-based UViM COCO panoptic task.
"""

import big_vision.configs.common as bvcc
import ml_collections as mlc


def get_config(arg='res=512,patch_size=16'):
  """Config for training label compression on COCO-panoptic."""
  arg = bvcc.parse_arg(arg, res=512, patch_size=16,
                       runlocal=False, singlehost=False)
  config = mlc.ConfigDict()

  config.input = {}
  config.input.data = dict(name='coco/2017_panoptic', split='train[4096:]')

  config.input.batch_size = 1024
  config.input.shuffle_buffer_size = 25_000

  config.total_epochs = 500

  config.input.pp = (
      f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
      f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
      f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'value_range(-1, 1)|make_canonical|copy("labels","image")|keep("image")'
  )
  pp_eval = (
      f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
      f'resize({arg.res})|resize({arg.res},key="labels",method="nearest")|'
      f'value_range(-1, 1)|make_canonical|copy("labels","image")|keep("image", "image/id")'
  )

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None

  # Model section
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
  config.model.codeword_dim = 32
  config.model.code_dropout = 'none'
  config.model.bottleneck_resize = True
  config.model.scan = True
  config.model.remat_policy = 'nothing_saveable'

  config.rec_loss_fn = 'xent'  # xent, l2
  # values: (index in source image, number of classes)
  config.model.inout_specs = {
      'semantics': (0, 133 + 1),  # +1 for void label
      'instances': (1, 100),  # COCO: actually 98 train/78 validation.
  }

  config.beta = 2.5e-4
  config.beta_percept = 0.0

  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)
  config.grad_clip_norm = 1.0

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
  config.evals.val.data.split = 'train[:4096]'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 250

  base = {
      'type': 'proj.givt.coco_panoptic',
      'pp_fn': pp_eval,
      'log_steps': 5_000,
      'pred': 'predict_panoptic',
      # Filters objects that occupy less than 0.03^2 fraction of all pixels.
      # 'pred_kw': {'min_fraction': 0.03 ** 2},
  }
  config.evals.coco_panoptic_train = dict(**base, data={'split': 'train[4096:8192]'})
  config.evals.coco_panoptic_holdout = dict(**base, data={'split': 'train[:4096]'})
  config.evals.coco_panoptic = dict(**base, data={'split': 'validation'})

  config.evals.save_pred = dict(type='proj.givt.save_predictions')
  config.evals.save_pred.pp_fn = pp_eval
  config.evals.save_pred.log_steps = 100_000
  config.evals.save_pred.pred = 'predict_panoptic'
  config.evals.save_pred.data = {**config.input.data}
  config.evals.save_pred.data.split = 'validation[:1024]'
  config.evals.save_pred.outfile = 'inference.npz'

  config.seed = 0

  if arg.singlehost:
    config.input.batch_size = 128
    config.num_epochs = 100
  elif arg.runlocal:
    config.input.batch_size = 16
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.enc_depth = 1
    config.model.dec_depth = 1

  return config