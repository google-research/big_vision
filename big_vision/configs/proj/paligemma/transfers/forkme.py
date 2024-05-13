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
r"""Example config for finetuning PaliGemma to a task stored in the JSON-L file, designed to fit on four L4 GPU.

Can be used as a starting point to finetune PaliGemma model. If you prefer to 
use tfds-based data input, check out other transfer configs as examples.

Command to run this config:

```
env BV_GEMMA_DIR=ckpts/ python -m big_vision.trainers.proj.paligemma.train \
    --config big_vision/configs/proj/paligemma/transfers/forkme.py \
    --workdir workdirs/`date '+%m-%d_%H%M'`
```
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER


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
      combine_and_keep_train(text_len),
  ])
  # Keep the whole dataset in RAM after first pass. Useful optimization for
  # small/mid-size datasets, but risks a host OOM for large datasets.
  c.cache_raw = True
  return c


def add_eval_pplx(c, res, text_len):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_data = training_data(res, text_len)  # Use mostly same settings as training.
  c_data.pp = '|'.join([
      # Read and prepare the image by just resizing it:
      f'decode|resize({res}, antialias=True)|value_range(-1, 1)',
      # The texts are already prepared in `prefix` and `suffix` keys.
      'strfmt("caption en", outkey="prefix")',
      combine_and_keep_eval(text_len),
  ])

  c.evals['val/pplx'] = dict(
      type='proj.paligemma.perplexity', pred='logits',
      key='text', shift_labels=True,
      log_percent=1/10,
      data={**c_data.data,
            'fname': 'gs://longcap100/data_val10.jsonl',
            },
      pp_fn=c_data.pp
      )


def add_eval_store(c, res, text_len=32):
  """Captioning evaluator with cider/bleu-4/meteor/rouge/spice metrics."""
  c_data = training_data(res, text_len)  # Use mostly same settings as training.
  c_data.pp = '|'.join([
      # Read and prepare the image by just resizing it:
      f'decode|resize({res}, antialias=True)|value_range(-1, 1)',
      # The texts are already prepared in `prefix` and `suffix` keys.
      'strfmt("caption en", outkey="prefix")',
      combine_and_keep_eval(text_len, keep=('id',)),
  ])

  c.evals['val/store'] = dict(
      type='proj.paligemma.transfers.storepreds',
      pred='decode', pred_kw={'max_decode_len': text_len},
      log_percent=0.5, tokenizer=TOKENIZER,
      data={**c_data.data,
            'fname': 'gs://longcap100/data_val10.jsonl',
            },
      pp_fn=c_data.pp,
  )


def get_config(arg=None):
  """Config for training."""
  # You probably do NOT want to add settings here. The `arg` way of settings is
  # really only for things you'd want to sweep and which affect MULTIPLE config
  # settings at once or go into the pp string.
  c = bvcc.parse_arg(arg, res=224, text_len=128, batch_size=32,
                     freeze_vit=False, freeze_llm=False,
                     run_local=False)

  c.input = training_data(c.res, c.text_len)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  c.total_epochs = 15
  c.input.batch_size = c.batch_size
  c.optax_name = 'scale_by_adam'
  c.lr = 1e-5
  c.wd = 3e-7
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
  ]

  c.evals = {}
  add_eval_pplx(c, c.res, c.text_len)
  add_eval_store(c, c.res, c.text_len)

  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=True)
  c.model.llm = dict(vocab_size=256_000 + 1024 + 128, dropout=0.0)
  c.model_init = f'pt_{c.res}'

  # FSDP strategy.
  c.mesh = [('data', -1)]
  c.sharding_strategy = [('.*', 'fsdp(axis="data")')]
  c.sharding_rules = [('act_batch', ('data',))]

  c.input.shuffle_buffer_size = 1000
  c.log_training_steps = 1
  c.ckpt_steps = 200
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops']

  c.seed = 0

  return c
