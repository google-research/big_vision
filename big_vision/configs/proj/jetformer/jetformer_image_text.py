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
r"""Train JetFormer (https://arxiv.org/abs//2411.19722) on image-text data.

Example launch command (local; see main README for launching on TPU servers):

  python -m big_vision.trainers.proj.jetformer.train \
    --config big_vision/configs/proj/jetformer/jetformer_image_text.py \
    --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%m-%d_%H%M'`

Add the suffix `:key1=value1,key2=value2,...` to the config path in the launch
command to modify the the config with the arguments below.
"""

import itertools

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
    '750M': {
        'width': 1536,
        'depth': 24,
        'mlp_dim': 1536 * 4,
        'num_heads': 16,
        'num_kv_heads': 1,
        'head_dim': 96,
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
    '350M': (9, 0.96),
    '750M': (8, 0.94),
    '1p3B': (6, 0.96),
}

TOKENIZER = 'sp(\'c4_en\')'
EOS_ID = 1


def get_config(arg=None):
  """A config for training GIVT + NVP on WebLI."""
  config = bvcc.parse_arg(
      arg, res=256, patch_size=16, model_size='350M', text_len=64,
      use_adaptor=True, depth_to_seq=1, use_boi=True,
      runlocal=False, num_replicas=1, num_slices=1, ignore_pad=False)

  config.pp_modules = (
      'ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops')

  # Data section
  pp_image = (
      f'decode'
      f'|resize_small({config.res}, inkey="image", outkey="image",'
      f'method="bicubic", antialias=True)'
      f'|central_crop({config.res})'
      f'|value_range(-1, 1, key="image")')

  def tokenize(inkey):
    """Tokenizes inkey into "text" and "text_mask" and "text_loss"."""
    # `text_mask` is 1 if tokens are allowed to attend to it.
    # `text_loss` is 1 if predicting it is part of the loss.
    # This has two modes:
    # 1. `ignore_pad=False`:
    #    - Model learns [BOS, A, B, C, EOS, pad..., BOI?, img1, img2, ...]
    #    - `text_mask` and `text_loss` are 1 for all tokens.
    # 2. `ignore_pad=True`:
    #    - Model learns [BOS, A, B, C, EOS, img1, img2, ...]
    #    - `text_mask` = 1 for text tokens, 0 for EOS and padding.
    #    - `text_loss` = 1 for text tokens and EOS and 0 for padding.
    ip = 0 if config.ignore_pad else 1
    return '|'.join([
        f'copy(inkey="{inkey}", outkey="text")',
        'lower(key="text")',
        f'tok(model="{TOKENIZER}", eos="no", key="text")',
        f'setdefault("eos", [{EOS_ID}])',
        f'masked_concat(["text", "eos"], outkey="text", text_loss=[1, 1], text_mask=[1, {ip}])',
        f'tolen(length={config.text_len}, pad_value={ip}, key="text")',
        f'tolen(length={config.text_len}, pad_value={ip}, key="text_mask")',
        f'tolen(length={config.text_len}, pad_value={ip}, key="text_loss")',
    ])
  vocab_size = 32_000

  config.input = {}
  # Using COCO captions as a dummy dataset since it is available on TFDS
  # without manual download. This won't produce good results - please use
  # your preferred large image-text data set.
  config.input.data = dict(name='coco_captions', split='train')
  pp_text = '|'.join([
      'choice(inkey="captions/text", outkey="text")',
      tokenize('text'),
  ])

  config.input.pp = f'{pp_image}|flatten|{pp_text}|keep("image", "text", "text_mask", "text_loss")'

  # Steps, batch size and friends.
  config.total_steps = 500_000
  config.input.batch_size = 4096
  config.input.shuffle_buffer_size = 25_000

  config.log_training_steps = 50
  config.ckpt_steps = 500
  config.keep_ckpt_percent = 0.0

  # Sample config.
  cfg_w, temp = SAMPLING_PARAMS[config.model_size]
  config.sample_images = {}
  config.sample_images.cfg_inference_weight = cfg_w
  config.sample_images.temperature = temp
  config.sample_images.temperature_probs = 1.0

  # Patch PCA section.
  # Does not apply PCA, but only splits image into patches and adds
  # dequantization noise.
  config.patch_pca = {}
  config.patch_pca.model_name = 'proj.jetformer.patch_pca'
  config.patch_pca.model = ml_collections.ConfigDict()
  config.patch_pca.model.depth_to_seq = config.depth_to_seq
  config.patch_pca.model.input_size = (config.res, config.res)
  config.patch_pca.model.patch_size = (config.patch_size, config.patch_size)
  config.patch_pca.model.code_len = ((config.res // config.patch_size) ** 2) * config.depth_to_seq
  config.patch_pca.model.codeword_dim = 128 // config.depth_to_seq
  config.patch_pca.model.noise_std = 0.0
  config.patch_pca.model.add_dequant_noise = True
  config.patch_pca.model.skip_pca = True

  # Transformer section.
  config.model_name = 'proj.jetformer.jetformer'
  config.model_init = ''
  assert config.model_size in GIVT_MODELS, f'Unknown model size: {config.model_size}'
  config.model = ml_collections.ConfigDict(GIVT_MODELS[config.model_size])
  # Alloc special tokens
  token_id = itertools.count(vocab_size)
  config.model.bos_id = next(token_id)
  if config.ignore_pad:
    assert config.use_boi, 'Ignore padding uses EOS as BOI'
    config.model.boi_id = 1
  else:
    if config.use_boi:
      config.model.boi_id = next(token_id)
  config.model.nolabel_id = next(token_id)
  config.model.vocab_size = next(token_id)
  config.model.out_dim = config.patch_pca.model.codeword_dim
  config.model.num_mixtures = 1024
  config.model.scale_tol = 1e-6
  config.model.drop_labels_probability = 0.1
  config.model.dropout = 0.1
  config.model.head_dtype = 'bfloat16'  # Seems stable and save memory.
  # Required for model sharding
  config.model.scan = True
  config.model.remat_policy = 'nothing_saveable'
  config.model.untie_output_vocab = True
  config.model.causal_mask_on_prefix = True

  config.input_noise_std = 0.3
  config.noise_scale = 64.0
  config.noise_min = 3.0
  config.rgb_noise_on_image_prefix = False
  config.latent_noise_dim = (16 * 16 * 3) - config.patch_pca.model.codeword_dim

  config.text_prefix_prob = 0.5
  config.loss_on_prefix = False
  config.text_loss_weight = 0.0025
  config.stop_grad_nvp_prefix = True

  # NVP section.
  config.adaptor_name = 'proj.jet.jet' if config.use_adaptor else ''
  config.adaptor = {}
  config.adaptor.model = {}
  config.adaptor.model.depth = 32
  config.adaptor.model.block_depth = NVP_BLOCK_DEPTH[config.model_size]
  config.adaptor.model.emb_dim = 512
  config.adaptor.model.num_heads = 8
  config.adaptor.model.ps = 1
  config.adaptor.model.kinds = ('channels',)
  config.adaptor.model.channels_coupling_projs = ('random',)
  config.adaptor.model.spatial_coupling_projs = ('checkerboard', 'checkerboard-inv',)

  config.optax_name = 'scale_by_adam'
  config.optax = dict(b2=0.95)
  config.grad_clip_norm = 1.0

  # FSDP training by default.
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

  # Standard schedule.
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

  ### Evaluation section.
  config.evals = {}
  config.evals.val = dict(
      type='mean',
      pred='validation',
      data={**config.input.data, 'split': 'val[:4096]'},
      pp_fn=config.input.pp,
      log_steps=5_000,
  )

  # config.evals.sample_images = dict(
  #     type='proj.givt.save_predictions',
  #     pp_fn=config.input.pp + '|keep("text","text_mask")',
  #     log_percent=0.1,
  #     pred='sample_images',
  #     data={**config.input.data, 'split': 'train[:128]'},
  #     outfile='inference_sampled_images.npz',
  #     skip_first=True,
  # )

  config.seed = 0
  config.ckpt_timeout = 30

  if config.runlocal:
    config.input.batch_size = 8
    config.input.shuffle_buffer_size = 10
    config.log_training_steps = 5
    config.model.depth = 2
    if config.use_adaptor:
      config.adaptor.model.depth = 1
    for k in config.evals.keys():
      if not hasattr(config.evals[k], 'data'):
        continue
      config.evals[k].data.split = config.evals[k].data.split.split('[')[0] + '[:16]'

  return config
