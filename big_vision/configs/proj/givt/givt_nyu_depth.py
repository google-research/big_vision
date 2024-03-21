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
r"""Train a GIVT encoder-decoder model for NYU depth prediction."""

import itertools
import big_vision.configs.common as bvcc
import ml_collections

ConfigDict = ml_collections.ConfigDict

VTT_MODELS = {
    'base': dict(num_layers=12, num_decoder_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768),
    'large': dict(num_layers=24, num_decoder_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024),
}

RES = 512
PATCH_SIZE = 16
LABEL_RES = 512
LABEL_PATCH_SIZE = 16
QUANTIZATION_BINS = 256
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0


def get_config(arg='split=sweep'):
  """Config for training."""
  arg = bvcc.parse_arg(arg, split='sweep', runlocal=False, singlehost=False)
  config = ConfigDict()

  config.input = {}
  config.input.pp = (
      f'decode|nyu_depth|'
      f'randu("fliplr")|det_fliplr(key="image")|det_fliplr(key="labels")|'
      f'inception_box|crop_box(key="image")|crop_box(key="labels")|'
      f'resize({RES})|'
      f'resize({LABEL_RES},key="labels",method="nearest")|'
      f'bin_nyu_depth(min_depth={MIN_DEPTH}, max_depth={MAX_DEPTH}, num_bins={QUANTIZATION_BINS})|'
      f'value_range(-1,1)|'
      f'copy("image", "cond_image")|copy("labels", "image")|'
      f'keep("image", "cond_image")'
  )
  pp_eval = (
      f'decode|nyu_depth|'
      f'nyu_eval_crop|'
      f'resize({RES})|'
      f'resize({LABEL_RES},key="labels",method="nearest")|'
      f'bin_nyu_depth(min_depth={MIN_DEPTH}, max_depth={MAX_DEPTH}, num_bins={QUANTIZATION_BINS})|'
      f'value_range(-1,1)|'
      f'copy("image", "cond_image")|copy("labels", "image")|'
      f'keep("image", "cond_image")'
  )
  pp_predict = (
      f'decode|nyu_depth|'
      f'nyu_eval_crop|copy("labels","ground_truth")|'
      f'resize({RES})|'
      f'value_range(-1,1)|'
      f'copy("image", "cond_image")|'
      f'strong_hash(inkey="tfds_id", outkey="image/id")|'
      f'keep("cond_image", "ground_truth", "image/id")'
  )

  config.input.data = dict(name='nyu_depth_v2', split='train')
  config.input.batch_size = 512
  config.input.shuffle_buffer_size = 50_000

  config.total_epochs = 50

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None
  config.prefetch_to_device = 2
  config.seed = 0

  # Optimizer section
  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)

  config.ar_generation_config = ConfigDict()
  config.ar_generation_config.temp = 0.9
  config.ar_generation_config.temp_probs = 1.0
  config.ar_generation_config.beam_size = 2
  config.ar_generation_config.fan_size = 8
  config.ar_generation_config.rand_top_k = False
  config.ar_generation_config.rand_top_k_temp = 1.0

  config.lr = 0.001
  config.wd = 0.000001
  config.lr_mults = [
      ('pos_embedding_encoder.*', 0.1),
      ('EmbedPatches.*', 0.1),
      ('encoder.*', 0.1),
      ('decoder.*', 1.0)
  ]
  config.schedule = dict(decay_type='cosine', warmup_percent=0.1)

  # Oracle section
  config.min_depth = MIN_DEPTH
  config.max_depth = MAX_DEPTH
  config.vae = ConfigDict()
  config.vae.model_name = 'proj.givt.vit'
  config.vae.model = ConfigDict()
  config.vae.model.input_size = (RES, RES)
  config.vae.model.patch_size = (PATCH_SIZE, PATCH_SIZE)
  config.vae.model.code_len = 256
  config.vae.model.width = 768
  config.vae.model.enc_depth = 6
  config.vae.model.dec_depth = 12
  config.vae.model.mlp_dim = 3072
  config.vae.model.num_heads = 12
  config.vae.model.codeword_dim = 16
  config.vae.model.code_dropout = 'none'
  config.vae.model.bottleneck_resize = True
  # values: (channel index in source image, number of classes)
  config.vae.model.inout_specs = {
      'depth': (0, QUANTIZATION_BINS),
  }
  config.vae.model_init = 'gs://big_vision/givt/vae_nyu_depth_params.npz'

  # Model section
  config.model_name = 'proj.givt.givt'
  # # Base model (for exploration)
  # config.model_init = {'encoder': 'howto-i21k-B/16'}
  # config.model = ConfigDict(VTT_MODELS['base'])
  # Large model
  config.model_init = {'encoder': 'howto-i21k-L/16'}
  config.model_load = dict(dont_load=('cls', 'head/bias', 'head/kernel'))
  config.model = ConfigDict(VTT_MODELS['large'])
  config.model.patches = (PATCH_SIZE, PATCH_SIZE)
  config.model.input_size = (RES, RES)
  config.model.posemb_type = 'learn'
  config.model.seq_len = config.vae.model.code_len
  config.model.num_labels = None
  config.model.num_mixtures = 1
  config.model.fix_square_plus = True
  config.model.out_dim = config.vae.model.codeword_dim
  config.model.scale_tol = 1e-6
  config.model.dec_dropout_rate = 0.0

  # Evaluation section
  config.evals = {}
  config.evals.val = ConfigDict()
  config.evals.val.type = 'mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = {**config.input.data}
  config.evals.val.data.split = 'validation'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 250

  base = {
      'type': 'proj.givt.nyu_depth',
      'data': {**config.input.data},
      'pp_fn': pp_predict,
      'pred': 'sample_depth',
      'log_steps': 2000,
      'min_depth': MIN_DEPTH,
      'max_depth': MAX_DEPTH,
  }

  config.evals.nyu_depth_val = dict(base)
  config.evals.nyu_depth_val.data.split = 'validation'

  config.evals.save_pred = dict(base)
  config.evals.save_pred.type = 'proj.givt.save_predictions'
  del config.evals.save_pred.min_depth, config.evals.save_pred.max_depth
  config.evals.save_pred.log_steps = 100_000
  config.evals.save_pred.data.split = 'validation[:128]'
  config.evals.save_pred.outfile = 'inference.npz'

  config.eval_only = False
  config.seed = 0

  if arg.runlocal:
    config.input.batch_size = 4
    config.input.shuffle_buffer_size = 10
    config.evals.val.log_steps = 20
    config.evals.val.data.split = 'validation[:4]'
    config.evals.nyu_depth_val.data.split = 'validation[:4]'
    config.evals.save_pred.data.split = 'validation[:4]'
    config.model.update(VTT_MODELS['base'])
    del config.model_init
    for k in config.evals.keys():
      if k not in ['val', 'nyu_depth_val', 'save_pred']:
        del config.evals[k]

  return config
