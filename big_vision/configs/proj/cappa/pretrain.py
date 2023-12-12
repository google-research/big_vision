# Copyright 2023 Big Vision Authors.
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
r"""Trains a CapPa model (https://arxiv.org/abs/2306.07915) on coco_captions.

This config is for reference, we never ran a full training on a large
image/text data set on public infrastructure.

big_vision.trainers.proj.cappa.generative \
  --config big_vision/configs/proj/cappa/pretrain.py \
  --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'`
"""


from big_vision.configs import common_fewshot
import big_vision.configs.common as bvcc
import ml_collections


def get_config(arg=None):
  """Returns the base config."""
  config = bvcc.parse_arg(arg,
                          runlocal=False,
                          total_steps=366_500,
                          batch_size=8*1024,
                          warmup_steps=10_000,
                          )

  config.evals = {}
  config.input = {}
  config.input.batch_size = config.batch_size if not config.runlocal else 8
  shuffle_buffer_size = 50_000 if not config.runlocal else 50

  res = 224
  patch_size = 16
  max_text_tokens = 64

  pp_image = (f'resize({res})|value_range(-1,1)')

  def tokenizer(inkey, outkey):
    return (f'tokenize(max_len={max_text_tokens}, model="c4_en", '
            f'eos="sticky", inkey="{inkey}", outkey="{outkey}")')

  pp_coco = (f'decode|{pp_image}|'
             'coco_captions("captions")|choice(inkey="captions", outkey="text")|'
             f'{tokenizer("text", "labels")}|keep("image", "labels")')
  config.input.pp = pp_coco

  # NOTE: "coco_captions" is way too small a dataset to train on. It's simply
  # used here to serve as a smoke test that the implementation works correctly.
  config.input.data = dict(name='coco_captions', split='train')  # num_examples=82_783
  config.input.shuffle_buffer_size = shuffle_buffer_size

  config.evals.val_coco = {
      'type': 'proj.cappa.perplexity',
      'pred': 'perplexity',
      'log_steps': 1000,
      'data': dict(name='coco_captions', split='val'),  # num_examples=5_000
      'pp_fn': pp_coco,
  }

  # Few-shot  metrics
  config.evals.fewshot = common_fewshot.get_fewshot_lsr(
      target_resolution=res, resize_resolution=int(256 / 224 * res))
  config.evals.fewshot.type = 'fewshot_lsr'
  config.evals.fewshot.log_steps = 5_000 if not config.runlocal else 5
  config.evals.fewshot.representation_layer = 'pre_logits'
  config.evals.fewshot.pred = 'enc_rep'
  config.evals.fewshot.pp_eval = config.evals.fewshot.pp_train

  # NOTE: Scoring of the entire imagenet validation set is rather slow:
  # ~100 secs / 1k classes / host.
  config.evals['imagenet/scoring'] = dict(
      type='proj.cappa.scoring_classifier',
      pred='score',
      log_percent=0.1,
      data=dict(name='imagenet2012', split='validation'),
      pp_fn=f'decode|{pp_image}|keep("image", "label")',
      pp_txt=tokenizer('label', 'labels'),
  )

  for e in config.evals.values():
    e.skip_first = True

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = None  # 10_000

  # Model section
  config.model_name = 'proj.cappa.cappa'
  config.model = ml_collections.ConfigDict()
  config.model.num_layers = 12
  config.model.num_heads = 12
  config.model.mlp_dim = 3072
  config.model.emb_dim = 768
  config.model.vocab_size = 32_000
  config.model.patches = (patch_size, patch_size)
  config.model.seq_len = max_text_tokens
  config.model.posemb_type = 'learn'

  # Decoder
  config.model.decoder_num_layers = 6
  # 0 values here mean to use the same value as for the encoder
  config.model.decoder_num_heads = 0
  config.model.decoder_mlp_dim = 0
  config.model.decoder_emb_dim = 0
  config.model.dec_dropout_rate = 0.0
  config.model.masked_pred_prob = 0.75
  config.model.masking_ratio = 1.0
  config.model.decoder_bias = False

  config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.999)
  config.grad_clip_norm = 1.0
  config.label_smoothing = 0.0

  schedule = dict(decay_type='cosine',
                  warmup_steps=config.warmup_steps
                  if not config.runlocal else 5)

  # Standard schedule
  config.lr = 0.001
  config.wd = 0.0001
  config.schedule = schedule

  config.seed = 0

  return config