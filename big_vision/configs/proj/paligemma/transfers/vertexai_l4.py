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
r"""PaliGemma transfer to a task stored in JSON-L, designed to fit on an L4 GPU.
"""

import big_vision.configs.common as bvcc


def training_data(res, text_len):
  """Creates training data config."""
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='bv:jsonl',
      fname='gs://longcap100/data_train90.jsonl',
      fopen_keys={'image': 'gs://longcap100/'},
      # See docstring in datasets/jsonl.py for further details.
      # download_keys=['image'],  # If jsonl contains external paths.
  )
  c.pp = '|'.join([
      # Read and prepare the image by just resizing it:
      f'decode|resize({res}, antialias=True)|value_range(-1, 1)',
      # The texts are already prepared in `prefix` and `suffix` keys.
      'strfmt("caption en", outkey="prefix")',
      combine_and_keep(text_len),
  ])
  # Keep the whole dataset in RAM after first pass. Useful optimization for
  # small/mid-size datasets, but risks a host OOM for large datasets.
  c.cache_raw = True
  return c


def get_config(arg=None):
  """Config for training."""
  # You probably do NOT want to add settings here. The `arg` way of settings is
  # really only for things you'd want to sweep and which affect MULTIPLE config
  # settings at once or go into the pp string.
  c = bvcc.parse_arg(arg, res=224, text_len=128, batch_size=4,
                     freeze_vit=False, freeze_llm=False)

  c.input = training_data(c.res, c.text_len)

  # These settings are suited for fitting in a single L4.
  c.total_epochs = 1
  c.input.batch_size = c.batch_size
  c.optax_name = 'big_vision.sgd'  # Without momentum, so really low-memory.
  c.lr = 0.1
  c.wd = 0.0
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
  ]

  c.evals = {}

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  # TODO: b/lbeyer - no scan and no remat might be better on 1-GPU machines?
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0)
  c.model_init = f'pt_{c.res}'

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  c.input.shuffle_buffer_size = 1000
  c.log_training_steps = 1
  c.ckpt_steps = 1_000
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  c.seed = 0
  return c


def tok(**kw):
  """Creates the tokenization preprocessing string."""
  # Single entry point so that it's consistent everywhere and easier to switch.
  kw.setdefault('model', 'gemma(tokensets=("loc", "seg"))')
  kw = ', '.join(f'{k}={repr(v)}' for k, v in kw.items())
  return f'tok({kw})'


def combine_and_keep(text_len):
  return '|'.join([
      tok(key='prefix', bos='yes'),
      tok(key='suffix', eos='yes'),
      tok(key='septok', text='\n'),
      # If masks confuse you, see (internal link)
      'masked_concat(["prefix", "septok", "suffix"], mask_ar=[0, 0, 1], mask_loss=[0, 0, 1])',
      # For training, we +1 because the trainer removes EOS.
      f'tolen({text_len+1}, pad_value=0, key="text")',  # For text, value doesn't matter.
      f'tolen({text_len+1}, pad_value=1, key="mask_ar")',
      f'tolen({text_len+1}, pad_value=0, key="mask_loss")',
      'keep("image", "text", "mask_ar", "mask_loss")',
  ])
