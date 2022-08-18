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
r"""A config for training a UViM stage II model for the panoptic task.

This config is expected to reproduce the paper's result and achieve
approximately 43.7 PQ points on the COCO holdout data.

We also provide a low-resource variant of this config, which can be enabled
by adding `:singlehost` postfix to the config name. This one is expected to
achieve 39.4 PQ points on the COCO holdout data.
"""

import big_vision.configs.common as bvcc
from ml_collections import ConfigDict

VTT_MODELS = {
    'base': dict(num_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768),
    'large': dict(num_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024),
}

VQVAE_MODELS = {
    'base': dict(enc_depth=6, dec_depth=12, num_heads=12, mlp_dim=3072, width=768),
}

RES = 512
PATCH_SIZE = 16
LABEL_RES = 512
LABEL_PATCH_SIZE = 16


def get_config(arg=''):
  """Config for training."""
  arg = bvcc.parse_arg(arg, runlocal=False, singlehost=False)
  config = ConfigDict()

  config.input = {}
  config.input.pp = (
      f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
      f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
      f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
      f'resize({LABEL_RES}, inkey="image", outkey="image_ctx")|'
      f'resize({RES})|resize({LABEL_RES},key="labels",method="nearest")|'
      f'value_range(-1, 1, key="image_ctx")|'
      f'value_range(-1, 1)|make_canonical|keep("image","image_ctx","labels")'
  )
  pp_eval = (
      f'decode|coco_panoptic|concat(["semantics","instances"], "labels")|'
      f'resize({LABEL_RES}, inkey="image", outkey="image_ctx")|'
      f'resize({RES})|resize({LABEL_RES},key="labels",method="nearest")|'
      f'value_range(-1, 1, key="image_ctx")|'
      f'value_range(-1, 1)|make_canonical|keep("image","image_ctx","labels")'
  )
  pp_predict = (
      f'resize({LABEL_RES}, inkey="image", outkey="image_ctx")|resize({RES})|'
      f'value_range(-1, 1, key="image_ctx")|value_range(-1, 1)|'
      f'keep("image","image_ctx","image/id")'  # image/id used for rng seeds.
  )

  config.input.data = dict(name='coco/2017_panoptic', split='train[4096:]')
  config.input.batch_size = 512
  config.input.shuffle_buffer_size = 50_000

  config.total_epochs = 200

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 5000
  config.prefetch_to_device = 2
  config.seed = 0

  # Optimizer section
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)

  config.lr = 0.001
  config.wd = 0.000001
  config.lr_mults = [
      ('pos_embedding_encoder.*', 0.1),
      ('EmbedPatches.*', 0.1),
      ('encoder.*', 0.1),
      ('decoder.*', 1.0)
  ]
  config.schedule = dict(decay_type='cosine', warmup_steps=4_000)

  # Oracle section
  config.oracle = ConfigDict()
  config.oracle.task = 'proj.uvim.panoptic_task'
  config.oracle.model_init = 'gs://big_vision/uvim/panoptic_stageI_params.npz'
  config.oracle.model_name = 'proj.uvim.vit'
  config.oracle.model = ConfigDict(VQVAE_MODELS['base'])
  config.oracle.model.input_size = (LABEL_RES, LABEL_RES)
  config.oracle.model.patch_size = (LABEL_PATCH_SIZE, LABEL_PATCH_SIZE)
  config.oracle.model.code_len = 256
  config.oracle.model.dict_size = 4096
  config.oracle.model.codeword_dim = 768
  config.oracle.model.with_encoder_ctx = True
  config.oracle.model.with_decoder_ctx = True
  config.oracle.model.code_dropout = 'random'
  config.oracle.model.bottleneck_resize = True
  config.oracle.model.inputs = {
      'semantics': (133 + 1, LABEL_PATCH_SIZE**2),  # +1 for void label
      'instances': (100, LABEL_PATCH_SIZE**2),  # COCO: actually 98 train/78 validation.
  }
  config.oracle.model.outputs = config.oracle.model.inputs

  # Model section
  config.model_name = 'proj.uvim.vtt'
  # config.model_init = {'encoder': 'howto-i21k-B/8'}
  config.model_init = {'encoder': 'howto-i21k-L/16'}
  config.model = ConfigDict(VTT_MODELS['large'])
  config.model.patches = ConfigDict({'size': (PATCH_SIZE, PATCH_SIZE)})
  config.model.vocab_size = config.oracle.model.get_ref('dict_size') + 1
  config.model.posemb_type = 'learn'
  config.model.input_size = (RES, RES)
  config.model.seq_len = config.oracle.model.get_ref('code_len')

  # Evaluation section
  config.evals = {}
  config.evals.val = ConfigDict()
  config.evals.val.type = 'proj.uvim.compute_mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = dict(name=config.input.data.name, split='train[:4096]')
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 1000

  base = {
      'type': 'proj.uvim.coco_panoptic',
      'pp_fn': pp_predict,
      'log_steps': 10_000,
      # Filters objects that occupy less than 0.03^2 fraction of all pixels.
      # 'predict_kwargs': {'min_fraction': 0.03 ** 2},
  }
  config.evals.coco_panoptic_train = dict(**base, split='train[4096:8192]')
  config.evals.coco_panoptic_holdout = dict(**base, split='train[:4096]')
  config.evals.coco_panoptic = dict(**base, split='validation')

  # config.evals.save_pred = dict(type='proj.uvim.save_predictions')
  # config.evals.save_pred.pp = pp_eval.replace('decode|', '')
  # config.evals.save_pred.log_steps = 100_000
  # config.evals.save_pred.dataset = config.dataset
  # config.evals.save_pred.split = 'validation[:1024]'
  # config.evals.save_pred.outfile = 'inference.npz'

  if arg.singlehost:
    config.input.batch_size = 32
    config.num_epochs = 50
  elif arg.runlocal:
    config.input.batch_size = 4
    config.input.shuffle_buffer_size = 10
    config.evals.val.data.split = 'train[:16]'
  return config