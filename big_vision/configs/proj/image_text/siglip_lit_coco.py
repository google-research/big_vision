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
r"""Minimal SigLIP (https://arxiv.org/abs/2303.15343) example.

Example training:

big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/lit_coco.py:batch_size=512 \
    --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%Y-%m-%d_%H%M'`
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, res=224, runlocal=False, token_len=16, txt='bert_base', img='B/16',
      init='', img_head=False, batch_size=512)
  img_name, img_init = common.inits[arg.img]
  txt_name, txt_init = common.inits[arg.txt]
  config = ConfigDict()

  config.input = {}
  config.input.data = dict(name='coco_captions', split='train')
  config.input.batch_size = arg.batch_size if not arg.runlocal else 32
  config.input.shuffle_buffer_size = 250_000  if not arg.runlocal else 50

  config.total_steps = 5_000 if not arg.runlocal else 1

  config.init_shapes = [(1, arg.res, arg.res, 3), (1, arg.token_len,)]
  config.init_types = ['float32', 'int32']

  if arg.init:
    vocab_path = arg.init.rsplit('.', 1)[0] + '.txt'
  else:
    vocab_path = f'{txt_init}/vocab.txt'
  tokenizer = lambda inkey: (
      f'bert_tokenize(inkey="{inkey}", max_len={arg.token_len}, '
      f'vocab_path="{vocab_path}")')
  config.input.pp = (
      f'decode|resize({arg.res})|flip_lr|randaug(2,10)|value_range(-1,1)'
      f'|flatten|{tokenizer("captions/text")}|keep("image", "labels")'
  )
  config.pp_modules = ['ops_general', 'ops_image', 'ops_text',
                       'proj.flaxformer.bert_ops', 'archive.randaug']

  config.log_training_steps = 50
  config.ckpt_steps = 1000

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
  config.model.bias_init = -2.71

  if txt_name == 'base':
    config.optax_name = 'scale_by_adam'
  else:
    config.optax_name = 'big_vision.scale_by_adafactor'

  config.lr = 0.001
  config.wd = 0.01
  warmup_steps = max(int(0.03 * config.total_steps), 100)
  config.schedule = [
      ('img/.*', None),  # Freezes image tower.
      ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps)),
  ]

  config.grad_clip_norm = 1.0

  config.evals = {}
  config.evals.retrieval_coco = common.get_coco(
      pp_img=f'resize({arg.res})|value_range(-1, 1)',
      pp_txt=tokenizer('texts'),
      log_steps=1000,
  )

  return config
