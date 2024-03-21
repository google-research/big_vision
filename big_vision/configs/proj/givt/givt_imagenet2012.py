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
r"""Train Generative Infinite Vocabulary Transformer (GIVT) on ImageNet.

Example launch command (local; see main README for launching on TPU servers):

  python -m big_vision.trainers.proj.givt.generative \
    --config big_vision/configs/proj/givt/givt_imagenet2012.py \
    --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%m-%d_%H%M'`

Add the suffix `:key1=value1,key2=value2,...` to the config path in the launch 
command to modify the the config with the arguments below. For example:
`--config big_vision/configs/proj/givt/givt_imagenet_2012.py:model_size=large`
"""

import big_vision.configs.common as bvcc
import ml_collections


RES = 256
PATCH_SIZE = 16

GIVT_MODELS = {
    'base': dict(num_decoder_layers=12, num_heads=12, mlp_dim=3072, emb_dim=768, dec_dropout_rate=0.1),
    'default': dict(num_decoder_layers=24, num_heads=16, mlp_dim=4096, emb_dim=1024, dec_dropout_rate=0.2),
    'large': dict(num_decoder_layers=48, num_heads=16, mlp_dim=8192, emb_dim=1536, dec_dropout_rate=0.3)
}


def get_config(arg=None):
  """A config for training a simple VAE on imagenet2012."""
  arg = bvcc.parse_arg(arg, res=RES, patch_size=PATCH_SIZE, style='ar',  # 'ar' or 'masked'
                       model_size='default', runlocal=False, singlehost=False,
                       adaptor=False)
  config = ml_collections.ConfigDict()

  config.input = {}
  ### Using Imagenette here to ensure this config is runnable without manual
  ### download of ImageNet. This is only meant for testing and will overfit
  ### immediately. Please download ImageNet to reproduce the paper results.
  # config.input.data = dict(name='imagenet2012', split='train[4096:]')
  config.input.data = dict(name='imagenette', split='train')

  config.input.batch_size = 8 * 1024 if not arg.runlocal else 8
  config.input.shuffle_buffer_size = 25_000 if not arg.runlocal else 10

  config.total_epochs = 500

  config.input.pp = (
      f'decode_jpeg_and_inception_crop({arg.res},'
      f'area_min=80, area_max=100, ratio_min=1.0, ratio_max=1.0,'
      f'method="bicubic", antialias=True)'
      f'|flip_lr'
      f'|value_range(-1, 1, key="image")'
      f'|copy("label", "labels")'
      f'|keep("image", "labels")')

  pp_eval = (
      f'decode'
      f'|resize_small({arg.res}, inkey="image", outkey="image",'
      f'method="bicubic", antialias=True)'
      f'|central_crop({arg.res})'
      f'|value_range(-1, 1, key="image")'
      f'|copy("label", "labels")'
      f'|keep("image", "labels")')

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None

  # Flags for AR model.
  config.ar_generation_config = ml_collections.ConfigDict()
  config.ar_generation_config.temp = 0.95
  config.ar_generation_config.temp_probs = 1.0
  config.ar_generation_config.beam_size = 1
  config.ar_generation_config.fan_size = 1
  config.ar_generation_config.rand_top_k = False
  config.ar_generation_config.rand_top_k_temp = 1.0
  config.ar_generation_config.cfg_inference_weight = 0.4

  # Flags for masked model.
  config.masked_generation_config = ml_collections.ConfigDict()
  config.masked_generation_config.choice_temperature = 35.0
  config.masked_generation_config.ordering = 'maskgit'
  config.masked_generation_config.cfg_inference_weight = 0.0
  config.masked_generation_config.schedule = 'cosine'

  # Used for eval sweep.
  config.eval_only = False

  # VAE section
  config.vae = {}
  config.vae.model = ml_collections.ConfigDict()
  config.vae.model.code_len = (arg.res // arg.patch_size) ** 2
  config.vae.model_name = 'proj.givt.cnn'
  config.vae.model.codeword_dim = 16
  config.vae.model.filters = 128
  config.vae.model.num_res_blocks = 2
  config.vae.model.channel_multipliers = (1, 1, 2, 2, 4)
  config.vae.model.conv_downsample = False
  config.vae.model.activation_fn = 'swish'
  config.vae.model.norm_type = 'GN'
  if arg.model_size == 'large':
    config.vae.model_init = 'gs://big_vision/givt/vae_imagenet_2012_beta_1e-5_params'
  else:
    config.vae.model_init = 'gs://big_vision/givt/vae_imagenet_2012_beta_5e-5_params'
  config.vae.model.malib_ckpt = True
  config.vae.model_load = {}
  config.vae.model_load.malib_ckpt = config.vae.model.malib_ckpt
  config.vae.model_load.use_ema_params = True

  # GIVT section
  config.model_name = 'proj.givt.givt'
  config.model_init = ''
  assert arg.model_size in GIVT_MODELS, f'Unknown model size: {arg.model_size}'
  config.model = ml_collections.ConfigDict(GIVT_MODELS[arg.model_size])
  config.model.num_layers = 0
  config.model.num_labels = 1000  # None
  config.model.seq_len = config.vae.model.code_len
  config.model.out_dim = config.vae.model.codeword_dim
  config.model.num_mixtures = 16
  config.model.posemb_type = 'learn'
  config.model.scale_tol = 1e-6
  config.model.style = arg.style
  config.model.min_masking_rate_training = 0.3
  config.model.mask_style = 'concat'
  config.model.drop_labels_probability = 0.1
  config.model.fix_square_plus = True
  config.model.per_channel_mixtures = False
  config.model_init = ''
  # Required for model sharding
  config.model.scan = True
  config.model.remat_policy = 'nothing_saveable'

  # Adaptor section
  config.adaptor_name = 'proj.givt.adaptor' if arg.adaptor else ''
  config.adaptor = {}
  config.adaptor.model = ml_collections.ConfigDict()
  config.adaptor.model.num_blocks = 8
  config.adaptor.model.num_channels_bottleneck = 4 * config.model.out_dim

  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)
  config.grad_clip_norm = 1.0

  # FSDP training by default
  config.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  config.sharding_rules = [('act_batch', ('data',))]

  # Standard schedule
  config.lr = 0.001
  config.wd = 0.0001
  config.schedule = dict(decay_type='cosine', warmup_percent=0.1)

  # MaskGIT-specific parameters
  if arg.style == 'masked':
    config.model.dec_dropout_rate = 0.4
    config.wd = 0.0
    if arg.res == 512:
      config.masked_generation_config.choice_temperature = 140
  # GIVT-Causal 512px specific parameters
  elif arg.res == 512 and arg.model_size == 'large':
    config.model.dec_dropout_rate = 0.1
    # Set up space-to-depth/pixel shuffle
    config.vae.model.code_len //= 2
    config.vae.model.pixel_shuffle_patch_size = (1, 2)
    config.model.seq_len //= 2
    config.model.out_dim = config.vae.model.codeword_dim * 2
    config.model.num_mixtures = 32
    config.adaptor.model.num_channels_bottleneck = 8 * config.model.out_dim
    config.adaptor.model.pixel_shuffle_patch_size = (1, 2)
    # Update sampling config
    config.ar_generation_config.temp = 0.9
    config.ar_generation_config.cfg_inference_weight = 0.9

  ### Evaluation section
  config.evals = {}
  config.evals.val = ml_collections.ConfigDict()
  config.evals.val.type = 'mean'
  config.evals.val.pred = 'validation'
  config.evals.val.data = {**config.input.data}
  config.evals.val.data.split = f'train[:{4096 if not arg.runlocal else 8}]'
  config.evals.val.pp_fn = pp_eval
  config.evals.val.log_steps = 1_000 if not arg.runlocal else 20

  config.evals.save_pred_sampling = dict(
      type='proj.givt.save_predictions',
      pp_fn=pp_eval,
      log_steps=10_000,
      pred='sample',
      batch_size=512,
      data=dict(name=config.input.data.name, split='validation[:512]'),
      outfile='inference_sampled.npz',
  )

  config.seed = 0

  config.ckpt_timeout = 30

  if arg.runlocal:
    config.input.batch_size = 4
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.num_decoder_layers = 2

    config.evals.val.data.split = 'validation[:16]'
    config.evals.val.log_steps = 20

  return config
