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
r"""PaliGemma transfer to ActivityNet Video QA.

IMPORTANT: This config is based on an unreleased version of DeepMind Video
Readers (DMVR). Users can either set up DMVR using the open source code from
GitHub (see below for details), or add their own data loader of choice.
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER

TEXT_LEN = 64
DATASET_NAME = 'activitynet_qa'
# Numbers might need to be updated due to wipeout. Current from 2024-04-28
SPLIT_SIZE = {'train': 27610, 'valid': 15760, 'test': 6900}


def training_data(res, *, final_split, num_frames, stride):
  """Creates training data config.

  Args:
    res: The requested image resolution (eg 224).
    final_split: Train on all train+valid data.
    num_frames: number of sampled frames per video.
    stride: stride at which the frames are sampled.

  Returns:
    The ConfigDict for the input section.
  """
  pp = '|'.join([
      # prepare the frames by decoding, resizing, replicating, sampling:
      f'video_decode({res})|video_replicate_img({num_frames},{num_frames})',
      f'video_ensure_shape("image", {(num_frames, res, res, 3)})',
      # only one question/answer per example.
      'reshape([], key="question")|reshape([], key="answer")',
      'strfmt("answer en {question}", outkey="prefix")',
      'copy("answer", "suffix")',
      combine_and_keep_train(TEXT_LEN),
  ])

  c = bvcc.parse_arg('')
  c.data = {}
  splits = ['train', 'valid'] if final_split else ['train']
  raise NotImplementedError('Please implement a video reader of choice!')
  # For example DMVR https://github.com/google-deepmind/dmvr
  # The reader should support the following arguments:
  # - name: Name of the reader.
  # - dataset_name: Name of the data set.
  # - split: Data set split.
  # - num_frames: Number of frames sampled from the video.
  # - stride: Stride at which the video frames are sampled.
  # - deterministic_fs: Whether to sample the frames starting at the first
  #   frame or whether an offest should be chosen at random (if there are more
  #   frames than num_frames * stride)
  # - first_k_shards: Whether to only use the first k shards of the data
  #   (optional but useful for speeding up intermediate evaluations).
  for split in splits:
    c.data[split] = SPLIT_SIZE[split]
    c[split] = {'pp': pp}
    c[split].data = dict(
        # PLEASE ADD YOUR READER HERE:
        name='<add_your_data_loader_here>',
        dataset_name=DATASET_NAME, split=split,
        num_frames=num_frames, stride=stride,
        deterministic_fs=False)
  return c


def add_eval(c, res, num_frames, stride):  # pylint: disable=unused-argument
  """QA evaluator."""
  c_train = training_data(res, final_split=True, num_frames=num_frames, stride=stride)

  pp = '|'.join([
      # prepare the frames by decoding, resizing, replicating, sampling:
      f'video_decode({res})|video_replicate_img({num_frames},{num_frames})',
      f'video_ensure_shape("image", {(num_frames, res, res, 3)})',
      # only one question/answer per example.
      'reshape([], key="question")|reshape([], key="answer")',
      'strfmt("answer en {question}", outkey="prefix")',
      'strfmt("{id}#{example/video_id}: {question}", "question_id")',
      combine_and_keep_eval(TEXT_LEN, keep=('question_id', 'answer')),
  ])

  for freq, name, split, first_k_shards, skip_first_eval in [
      (1/8, 'minitrain', 'train', 2, False),  # To gauge memorization.
      (1/4, 'minival', 'valid', 2, False),  # To monitor val progress.
      (1, 'val', 'valid', None, True),  # To tune hparams.
      (1, 'eval', 'test', None, True),  # final metric
  ]:
    c.evals[f'activitynet_qa/{name}'] = dict(
        type='proj.paligemma.transfers.vqa',
        pred='decode', pred_kw={'max_decode_len': TEXT_LEN},
        data={**c_train.train.data, 'split': split,
              'first_k_shards': first_k_shards,
              'deterministic_fs': True},
        log_percent=freq, tokenizer=TOKENIZER,
        pp_fn=pp, skip_first=skip_first_eval)


def add_eval_pplx(c, res, num_frames, stride):
  """Perplexity evaluator to test runs before implementing the real deal."""
  c_train = training_data(res, final_split=True, num_frames=num_frames, stride=stride)

  for name, split, first_k_shards in [
      ('minitrain', 'train', 2),  # To gauge memorization.
      ('minival', 'valid', 2),
  ]:
    c.evals[f'activitynet_qa/{name}/pplx'] = dict(
        type='proj.paligemma.perplexity', pred='logits',
        key='text', shift_labels=True,
        log_percent=1/8,  # Not too cheap, do 10x per run.
        data={**c_train.train.data, 'split': split,
              'first_k_shards': first_k_shards,
              'deterministic_fs': True},
        pp_fn=c_train.train.pp,
    )


def sweep_best(add, arg=None):
  """Train with best hyper-params."""
  c = bvcc.parse_arg(arg, final_split=False)
  add(lr=1e-5, wd=1e-6, total_epochs=1, **bvcc.arg(num_frames=16, stride=70, res=224, **c))


sweep = sweep_best


def get_config(arg=None):
  """Config for training."""
  c = bvcc.parse_arg(arg, mode='xm', num_frames=16, stride=70, res=224,
                     freeze_vit=False, freeze_llm=False, final_split=False)

  c.input = training_data(
      c.res, final_split=c.final_split,
      num_frames=c.num_frames, stride=c.stride)

  c.total_epochs = 3
  c.input.batch_size = 128
  c.optax_name = 'scale_by_adam'
  c.optax = dict(b2=0.999)
  c.lr = 1e-5
  c.wd = 1e-6
  c.grad_clip_norm = 1.0
  c.label_smoothing = 0.0

  # Learning-rate schedule.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      ('img/.*', None if c.freeze_vit else sched),
      ('llm/.*', None if c.freeze_llm else sched),
  ]

  # Add evaluators.
  c.evals = {}
  add_eval(c, c.res, c.num_frames, c.stride)
  add_eval_pplx(c, c.res, c.num_frames, c.stride)

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

  for split in c.input.data.keys():
    c.input[split].shuffle_buffer_size = 10_000
  c.log_training_steps = 50
  c.ckpt_steps = 1_000
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops',
                  'proj.paligemma.video']

  # Update configs for quicker local runs and avoid swapping.
  if c.mode in ('runlocal', 'mock'):
    for split in c.input.data.keys():
      c.input[split].shuffle_buffer_size = None
    for ev in c.evals.values():
      ev.data.first_k_shards = 1

  if c.mode == 'runlocal':
    c.log_training_steps = 1
    c.input.batch_size = 2

  c.seed = 0
  return c


def metrics(arg=None):  # pylint: disable=unused-argument
  m = ['training_loss']
  for split in ('minitrain', 'minival', 'val', 'eval'):
    m.append(('epoch', f'{DATASET_NAME}/{split}/acc'))
  for split in ('minitrain', 'minival'):
    m.append(('epoch', f'{DATASET_NAME}/{split}/pplx/avg'))
  return m

