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
r"""A config for training a UViM stage II model for the depth task.
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
QUANTIZATION_BINS = 256
# Same as values used in eval, see evaluators/nyu_depth.py.
MIN_DEPTH = 1e-3
MAX_DEPTH = 10


def get_config(arg='split=final'):
  """Config for training."""
  arg = bvcc.parse_arg(arg, split='final', runlocal=False, singlehost=False)
  config = ConfigDict()

  config.input = {}
  config.input.pp = (
      f'decode|nyu_depth|'
      f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
      f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
      f'resize({RES})|'
      f'resize({LABEL_RES},inkey="image",outkey="image_ctx")|'
      f'resize({LABEL_RES},key="labels",method="nearest")|'
      f'value_range(-1,1)|'
      f'value_range(-1,1,inkey="image_ctx",outkey="image_ctx")|'
      f'keep("image","image_ctx","labels")'
  )
  pp_eval = (
      f'decode|nyu_depth|'
      f'nyu_eval_crop|'
      f'resize({RES})|'
      f'resize({LABEL_RES},inkey="image",outkey="image_ctx")|'
      f'resize({LABEL_RES},key="labels",method="nearest")|'
      f'value_range(-1,1)|'
      f'value_range(-1,1,inkey="image_ctx",outkey="image_ctx")|'
      f'keep("image","image_ctx","labels")'
  )
  pp_predict = (
      f'nyu_depth|'
      f'nyu_eval_crop|copy("labels","ground_truth")|'
      f'resize({RES})|'
      f'resize({LABEL_RES},inkey="image",outkey="image_ctx")|'
      f'value_range(-1,1)|'
      f'value_range(-1,1,inkey="image_ctx",outkey="image_ctx")|'
      f'keep("image","image_ctx","ground_truth")'
  )

  config.input.data = dict(name='nyu_depth_v2', split='train')
  config.input.batch_size = 512
  config.input.shuffle_buffer_size = 50_000

  config.total_epochs = 50

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 5000
  config.prefetch_to_device = 2
  config.seed = 0

  # Optimizer section
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)
  config.optax.clipping_threshold = None

  config.lr = 0.001
  config.wd = 0.000001
  config.lr_mults = (
      ('pos_embedding_encoder.*', 0.1),
      ('EmbedPatches.*', 0.1),
      ('encoder.*', 0.1),
      ('decoder.*', 1.0)
  )
  config.schedule = dict(decay_type='cosine', warmup_steps=4_000)

  # Oracle section
  config.oracle = ConfigDict()
  config.oracle.min_depth = MIN_DEPTH
  config.oracle.max_depth = MAX_DEPTH
  config.oracle.task = 'proj.uvim.depth_task'
  config.oracle.model_init = 'gs://big_vision/uvim/depth_stageI_params.npz'
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
      'depth': (QUANTIZATION_BINS, LABEL_PATCH_SIZE**2,),
  }
  config.oracle.model.outputs = config.oracle.model.inputs

  # Model section
  config.model_name = 'proj.uvim.vtt'
  # config.model_init = {'encoder': 'howto-i21k-B/8''}  # B/8 I21K
  config.model_init = {'encoder': 'howto-i21k-L/16'}  # L/16 I21K
  config.model = ConfigDict(VTT_MODELS['large'])
  config.model.patches = ConfigDict({'size': (PATCH_SIZE, PATCH_SIZE)})
  config.model.vocab_size = config.oracle.model.dict_size + 1
  config.model.posemb_type = 'learn'
  config.model.input_size = (RES, RES)
  config.model.seq_len = config.oracle.model.get_ref('code_len')
  config.model.zero_decoder_seq = False

  # Evaluation section
  config.evals = {}
  config.evals.val = ConfigDict()
  config.evals.val.type = 'proj.uvim.compute_mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = {**config.input.data}
  config.evals.val.data.split = 'validation'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 1000

  base = {
      'type': 'proj.uvim.nyu_depth',
      'dataset': config.input.data.name,
      'pp_fn': pp_predict,
      'log_steps': 2000,
      'min_depth': MIN_DEPTH,
      'max_depth': MAX_DEPTH,
  }
  config.evals.nyu_depth_val = dict(**base, split='validation')

  if arg.singlehost:
    config.input.batch_size = 32
    config.total_epochs = 20
  elif arg.runlocal:
    config.oracle.model_init = '/tmp/checkpoint.npz'
    config.model_init = {'encoder': '/tmp/enc_checkpoint.npz'}
    config.evals = {}
    config.input.batch_size = 1
    config.input.shuffle_buffer_size = 10
  return config