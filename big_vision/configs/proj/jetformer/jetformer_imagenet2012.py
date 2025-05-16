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
r"""Train JetFormer (https://arxiv.org/abs//2411.19722) on ImageNet256.

Example launch command (local; see main README for launching on TPU servers):

  python -m big_vision.trainers.proj.jetformer.train \
    --config big_vision/configs/proj/jetformer/jetformer_imagenet2012.py \
    --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%m-%d_%H%M'`

Add the suffix `:key1=value1,key2=value2,...` to the config path in the launch
command to modify the the config with the arguments below. For example:
`--config .../jetformer_imagenet2012.py:model_size=1p3B,total_epochs=500`
"""

import big_vision.configs.common as bvcc
import ml_collections

GIVT_MODELS = {
    '350M': {
        'width': 1024,
        'depth': 24,
        'mlp_dim': 1024 * 4,
        'num_heads': 16,
        'num_kv_heads': 1,
        'head_dim': 64,
    },
    '1p3B': {
        'width': 1536,
        'depth': 48,
        'mlp_dim': 1536 * 4,
        'num_heads': 16,
        'num_kv_heads': 1,
        'head_dim': 96,
    },
}

NVP_BLOCK_DEPTH = {
    '350M': 4,
    '1p3B': 6,
}

SAMPLING_PARAMS = {
    '350M': (3, 0.94),  # Tested with 100 and 500 epoch training.
    '1p3B': (2, 0.93),  # Tested with 500 epoch training.
}


def get_config(arg=None):
  """Train JetFormer on ImageNet."""
  config = bvcc.parse_arg(
      arg, res=256, patch_size=16, model_size='350M',
      total_epochs=100, use_adaptor=True, depth_to_seq=1,
      runlocal=False, num_replicas=1, num_slices=1)

  config.input = {}
  # Consider using Imagenette in case you don't want to download ImageNet.
  # (This won't produce good results, but will enable running the config)
  # config.input.data = dict(name='imagenette', split='train')
  config.input.data = dict(name='imagenet2012', split='train[4096:]')

  config.input.batch_size = 2048
  config.input.shuffle_buffer_size = 25_000

  config.input.pp = (
      f'decode'
      f'|resize_small({config.res}, inkey="image", outkey="image",'
      f'method="bicubic", antialias=True)'
      f'|central_crop({config.res})'
      f'|flip_lr'
      f'|value_range(-1, 1, key="image")'
      f'|reshape((1,), inkey="label", outkey="text")'
      f'|setdefault("text_loss", [1])'
      f'|copy("text_loss", "text_mask")'
      f'|keep("image", "text", "text_mask", "text_loss")')

  pp_eval = config.input.pp.replace('|flip_lr', '')

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None

  # Sample config
  cfg_w, temp = SAMPLING_PARAMS[config.model_size]
  config.sample_images = {}
  config.sample_images.cfg_inference_weight = cfg_w
  config.sample_images.temperature = temp
  config.sample_images.temperature_probs = 1.0

  sequence_length = ((config.res // config.patch_size) ** 2) * config.depth_to_seq

  # Patch PCA section.
  # Does not apply PCA, but only splits image into patches and adds
  # dequantization noise.
  config.patch_pca = {}
  config.patch_pca.model_name = 'proj.jetformer.patch_pca'
  config.patch_pca.model = ml_collections.ConfigDict()
  config.patch_pca.model.depth_to_seq = config.depth_to_seq
  config.patch_pca.model.input_size = (config.res, config.res)
  config.patch_pca.model.patch_size = (config.patch_size, config.patch_size)
  config.patch_pca.model.code_len = sequence_length
  config.patch_pca.model.codeword_dim = 128 // config.depth_to_seq
  config.patch_pca.model.noise_std = 0.0
  config.patch_pca.model.add_dequant_noise = True
  config.patch_pca.model.skip_pca = True

  # Transformer section.
  config.model_name = 'proj.jetformer.jetformer'
  config.model_init = ''
  config.model = ml_collections.ConfigDict(GIVT_MODELS[config.model_size])
  num_labels = 1000
  config.model.bos_id = num_labels
  config.model.boi_id = num_labels + 1
  config.model.nolabel_id = num_labels + 2
  config.model.vocab_size = num_labels + 3
  config.model.out_dim = config.patch_pca.model.codeword_dim
  config.model.num_mixtures = 1024
  config.model.scale_tol = 1e-6
  config.model.dropout = 0.1
  config.model.drop_labels_probability = 0.1
  config.model_init = ''
  config.model.head_dtype = 'bfloat16'
  # Required for model sharding
  config.model.scan = True
  config.model.remat_policy = 'nothing_saveable'
  config.model.num_vocab_repeats = 16

  # Noise on the latent representation after NVP.
  config.input_noise_std = 0.3
  # Noise curriculum.
  config.noise_scale = 64.0
  # Dimensionality of factored out features.
  config.latent_noise_dim = (16 * 16 * 3) - config.patch_pca.model.codeword_dim

  config.text_prefix_prob = 1.0
  config.loss_on_prefix = False
  # NVP section.
  config.adaptor_name = 'proj.jet.jet' if config.use_adaptor else ''
  config.adaptor = {}
  config.adaptor.model = dict(
      depth=32, block_depth=NVP_BLOCK_DEPTH[config.model_size],
      emb_dim=512, num_heads=8, ps=1,
      kinds=('channels',),
      channels_coupling_projs=('random',),
      spatial_coupling_projs=('checkerboard', 'checkerboard-inv'),)
  config.adaptor.class_conditional = False

  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)
  config.grad_clip_norm = 1.0
  config.ema_decay = 0.0

  # FSDP training by default
  config.sharding_strategy = [
      ('.*FREEZE_ME.*', 'replicate'),
      ('.*', 'fsdp(axis="fsdp")'),
  ]
  config.mesh = [
      ('slice', config.num_slices),
      ('replica', config.num_replicas),
      ('fsdp', -1),
  ]
  config.sharding_rules = [('act_batch', ('slice', 'replica', 'fsdp',))]

  # Standard schedule
  config.lr = 0.001
  config.wd = 0.0001
  config.wd_mults = [
      # Explicitly handling the Gemma kernels, which have different names.
      ('^decoder/layers/attn/.*', 1.0),
      ('^decoder/layers/mlp/.*', 1.0),
      ('.*/kernel$', 1.0),
  ]
  config.schedule = [
      ('.*FREEZE_ME.*', None),
      ('.*', dict(decay_type='cosine', warmup_percent=0.1)),
  ]

  ### Evaluation section
  config.evals = {}
  config.evals.val = dict(
      type='mean',
      pred='validation',
      data={**config.input.data, 'split': 'train[:4096]'},
      pp_fn=pp_eval,
      log_steps=1_000,
  )

  # config.evals.save_pred_sampling = dict(
  #     type='proj.givt.save_predictions',
  #     pp_fn=pp_eval,
  #     log_percent=0.1,
  #     pred='sample_images',
  #     batch_size=512,
  #     data=dict(name=config.input.data.name, split='validation[:512]'),
  #     outfile='inference_sampled.npz',
  #     pred_kw={'decode_len': sequence_length},
  # )

  config.seed = 0

  config.ckpt_timeout = 30

  if config.runlocal:
    config.input.batch_size = 8
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.depth = 2
    config.adaptor.model.depth = 1
    del config.evals.fid
    for k in config.evals.keys():
      if not hasattr(config.evals[k], 'data'):
        continue
      config.evals[k].data.split = config.evals[k].data.split.split('[')[0] + '[:16]'

  return config
