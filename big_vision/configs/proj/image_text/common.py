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

"""Snippets and constants used a lot in image-text configs."""

import ml_collections


# pylint: disable=line-too-long
inits = {
    # Downloaded & extracted from original repo:
    # https://github.com/google-research/bert
    'bert_base': ('base', 'gs://vit_models/lit/bert/uncased_L-12_H-768_A-12'),
    'bert_large': ('large', 'gs://vit_models/lit/bert/uncased_L-uncased_L-24_H-1024_A-16'),
    # Recommended "How to train your ViT..." checkpoints from
    # https://github.com/google-research/vision_transformer#available-vit-models
    'B/32': ('B/32', 'gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz'),
    'B/16': ('B/16', 'gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'),
    'L/16': ('L/16', 'gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz'),
}
# pylint: enable=line-too-long


def _square875(sz):
  return f'resize({int(sz/0.875)})|central_crop({sz})|value_range(-1,1)'


def _aspect75(sz):
  return f'resize_small({int(sz/0.75)})|central_crop({sz})|value_range(-1,1)'


def _drop_no_real_label(f):
  return len(f['real_label']) > 0


def _drop_no_imagenet(f):
  return len(f['labels_imagenet']) > 0


DISCLF_DATASET_OVERRIDES = {
    'imagenet2012': {'class_names': 'clip', 'split': 'validation'},
    'imagenet2012_minival': {
        'dataset_name': 'imagenet2012',
        'class_names': 'clip',
        'split': 'train[99%:]',
    },
    'imagenet2012_real': {
        'split': 'validation',
        'class_names': 'clip',
        'class_names_dataset_name': 'imagenet2012',
        'pp_img': lambda sz: (
            _square875(sz) + '|pad_to_shape(inkey="real_label", outkey="label", shape=[10], pad_value=-1)|keep("label", "image")'),  # pylint: disable=line-too-long
        'pre_filter_fn': _drop_no_real_label,
    },
    'imagenet_v2': {'class_names': 'clip'},
    'imagenet_a': {
        'class_names': 'clip',
        'pp_img': lambda sz: _aspect75(sz) + '|map("i1k_i1ka")',
    },
    'imagenet_r': {
        'class_names': 'clip',
        'pp_img': lambda sz: _square875(sz) + '|map("i1k_i1kr")',
    },
}


def get_disclf(sz, *, pp_txt=None, dataset_names=('imagenet2012',), **kw):
  """Returns config for discriminative_classifier of specified datasets."""
  config = ml_collections.ConfigDict(dict(
      dataset_names=list(dataset_names),
      type='proj.image_text.discriminative_classifier',
      prefix='z/0shot/',
      pp_img=_square875(sz),
      dataset_overrides={},
      cache_final=True,
      **kw,
  ))
  if pp_txt:
    config.pp_txt = pp_txt
  for name in dataset_names:
    if name in DISCLF_DATASET_OVERRIDES:
      config.dataset_overrides[name] = {**DISCLF_DATASET_OVERRIDES[name]}
      d = config.dataset_overrides[name]
      if 'pp_img' in d and callable(d['pp_img']):
        with d.ignore_type():
          d['pp_img'] = d['pp_img'](sz)
  return config


def get_coco(
    *,
    pp_img='resize(224)|value_range(-1, 1)',
    pp_txt='tokenize(max_len=16, inkey="texts", eos="sticky", pad_value=1)',
    prefix='z/retr/coco_',
    **kw):
  """Returns config for mscoco retrieval zero-shot.

  Args:
    pp_img: Pre-processing string for "image" feature.
    pp_txt: Pre-processing string for texts (expected to tokenize "texts" to
      "labels").
    prefix: Prefix to use for metrics.
    **kw: Other config settings, most notably log_{steps,percent,...}.

  Returns:
    `ConfigDict` that can be used as a retrieval evaluator configuration.
  """
  return ml_collections.ConfigDict({
      'type': 'proj.image_text.retrieval',
      'pp_txt': pp_txt,
      'pp_img': pp_img,
      'prefix': prefix,
      'dataset': 'coco_captions',
      'txt_name': ('captions', 'text'),
      **kw,
  })
