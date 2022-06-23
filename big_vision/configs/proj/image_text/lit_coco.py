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
r"""Trains a LiT model as in https://arxiv.org/abs/2111.07991

IMPORTANT NOTE: This config uses coco_captions for demonstration purposes. As of
6/17/22 neither YFCC100M nor CC12M are available in TFDS. We're working on
publishing these datasets to allow for full replication of the numbers reported
in the paper.

Published models:

https://github.com/google-research/vision_transformer#lit-models

Colab to load public LiT models:
https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

gs://vit_models/lit/LiT-B16B.npz - 72.07% i1k 0shot
gs://vit_models/lit/LiT-L16L.npz - 75.68% i1k 0shot - missing in publication

Example training:

big_vision.trainers.proj.image_text.contrastive \
    --config big_vision/configs/proj/image_text/lit_coco.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%Y-%m-%d_%H%M'`

Example evaluation:

big_vision.tools.eval_only \
    --config big_vision/configs/proj/image_text/lit_coco.py:txt=bert_base,img_head,img=B/16,init=gs://vit_models/lit/LiT-B16B.npz \
    --workdir gs://[your_bucket]/big_vision/`date '+%Y-%m-%d_%H%M'`
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common as cl
from big_vision.configs.proj.image_text import common_retrieval
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, res=224, runlocal=False, token_len=16, txt='bert_base', img='B/16',
      init='', img_head=False)
  img_name, img_init = cl.inits[arg.img]
  txt_name, txt_init = cl.inits[arg.txt]
  config = ConfigDict()

  config.batch_size = 4096*1 if not arg.runlocal else 32
  # TODO update config to use YFCC100M, CC12M from tfds
  config.dataset = 'coco_captions'
  config.train_split = 'train'
  config.total_steps = 5_000 if not arg.runlocal else 1

  config.init_shapes = [(1, arg.res, arg.res, 3), (1, arg.token_len,)]
  config.init_types = ['float32', 'int32']

  if arg.init:
    vocab_path = '.'.join(arg.init.split('.')[:-1]) + '.txt'
  else:
    vocab_path = f'{txt_init}/vocab.txt'
  tokenizer = lambda inkey: (
      f'bert_tokenize(inkey="{inkey}", max_len={arg.token_len}, '
      f'vocab_path="{vocab_path}")')
  config.pp_train = pp_eval = (
      f'decode|resize({arg.res})|flip_lr|randaug(2,10)|value_range(-1,1)'
      f'|flatten|{tokenizer("captions/text")}|keep("image", "labels")'
  )
  config.pp_modules = [
      'ops_general', 'ops_image', 'ops_text', 'proj.flaxformer.bert_ops']
  config.pp_img = f'resize({arg.res})|value_range(-1,1)|keep("image")'
  config.pp_txt = tokenizer('label') + '|keep("labels")'

  config.shuffle_buffer_size = 250_000  if not arg.runlocal else 50
  config.log_training_steps = 50
  config.checkpoint_steps = 1000

  # Model section
  config.model_name = 'proj.image_text.two_towers'
  config.model_load = {}
  if arg.init:
    config.model_init = arg.init
  else:
    config.model_init = {'image': img_init, 'text': txt_init}
    config.model_load['txt_load_kw'] = {'dont_load': ['head/kernel', 'head/bias']}
    if not arg.img_head:
      config.model_load['img_load_kw'] = {'dont_load': ['head/kernel', 'head/bias']}
  config.model = ConfigDict()
  config.model.image_model = 'vit'
  config.model.text_model = 'proj.flaxformer.bert'
  config.model.image = ConfigDict({
      'variant': img_name,
      'pool_type': 'tok',
      'head_zeroinit': False,
  })
  config.model.text = ConfigDict({
      'config': txt_name,
      'head_zeroinit': False,
  })
  config.model.temperature_init = 10.0
  dim = {'B': 768, 'L': 1024}[arg.img[0]]
  config.model.out_dim = (dim if arg.img_head else None, dim)  # (image_out_dim, text_out_dim)

  if txt_name == 'base':
    config.optax_name = 'scale_by_adam'
  else:
    config.optax_name = 'big_vision.scale_by_adafactor'
  # Gather representations across TPU cores for larger batch size for loss.
  # Generally helps: ((internal link)
  config.loss_use_global_batch = True

  config.lr = 0.001
  config.wd = 0.01
  warmup_steps = max(int(0.03 * config.total_steps), 100)
  config.schedule = [
      ('img/.*', None),
      ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps)),
  ]

  config.grad_clip_norm = 1.0
  # Gather representations across TPU cores for larger batch size for loss.
  # See Figure 9 from https://arxiv.org/abs/2111.07991
  config.loss_use_global_batch = True

  # Eval section (Both few-shot and zero-shot)
  eval_common = dict(
      type='proj.image_text.contrastive',
      use_global_batch=config.loss_use_global_batch,
      log_steps=500,
  )
  config.evals = {}
  sub = '[:4]' if arg.runlocal else ''
  config.evals.val = {
      **eval_common,
      'split': f'val{sub}',
      'dataset': config.dataset,
      'pp_fn': pp_eval,
  }
  config.evals.coco = {
      **eval_common,
      'dataset': 'coco_captions',
      'split': f'val{sub}',
      'pp_fn': (
          f'decode|resize({arg.res})|value_range(-1,1)'
          f'|flatten|{tokenizer("captions/text")}|keep("image", "labels")'),
  }
  config.evals.imagenet = {
      **eval_common,
      'dataset': 'imagenet2012',
      'split': f'validation{sub}',
      'pp_fn': (
          f'decode|resize({arg.res})|value_range(-1,1)'
          '|clip_i1k_label_names'
          f'|{tokenizer("labels")}|keep("image", "labels")'),
  }

  config.evals.disclf = {}
  config.evals.disclf.pp_img = f'resize({arg.res})|value_range(-1,1)'
  config.evals.disclf.pp_txt = tokenizer('texts')
  config.evals.disclf.type = 'proj.image_text.discriminative_classifier'
  config.evals.disclf.prefix = 'z/0shot/'
  config.evals.disclf.log_steps = eval_common['log_steps']
  config.evals.retrieval_coco = common_retrieval.get_coco(
      pp_img=f'resize({arg.res})|value_range(-1, 1)',
      pp_txt=tokenizer('texts'),
      log_steps=config.evals.disclf.log_steps,
  )

  config.seed = 0
  config.l = config.m = 0

  return config
