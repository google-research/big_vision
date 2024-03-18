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

"""Utilities for generating zero-shot prompts."""

import re
import string
from typing import Sequence

from absl import logging
from big_vision.datasets.imagenet import class_names as imagenet_class_names
from big_vision.evaluators.proj.image_text import prompt_engineering_constants
import tensorflow_datasets as tfds


_CLASS_NAMES = {  # For each dataset, maps from a source to its class names.
    "imagenet2012": {
        "clip": imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES,
    },
    "grand-vision:imagenet2012": {
        "clip": imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES,
    },
    "imagenet_a": {
        "clip": [
            imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES[i]
            for i in imagenet_class_names.IMAGENET_A_LABELSET
        ]
    },
    "imagenet_r": {
        "clip": [
            imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES[i]
            for i in imagenet_class_names.IMAGENET_R_LABELSET
        ]
    },
    "imagenet_v2": {
        "clip": imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES,
    },
}

_PROMPT_TEMPLATES = {
    "class_name_only": ["{}"],
    "clip_paper": prompt_engineering_constants.CLIP_PAPER_PROMPT_TEMPLATES,
    "clip_best": prompt_engineering_constants.CLIP_BEST_PROMPT_TEMPLATES,
}


def get_class_names(*, dataset_name, source="dataset_info", canonicalize=True):
  """Returns class name for `dataset_name` from `source`."""
  if isinstance(source, str):
    if source.startswith("dataset_info:"):
      name = source[len("dataset_info:"):]
      class_names = tfds.builder(dataset_name).info.features[name].names
    else:
      class_names = _CLASS_NAMES[dataset_name][source]
  else:
    assert isinstance(source, Sequence) and all(
        map(lambda s: isinstance(s, str), source)), source
    class_names = source
  if canonicalize:
    class_names = [
        canonicalize_text(name, keep_punctuation_exact_string=",")
        for name in class_names
    ]
  logging.info("Using %d class_names: %s", len(class_names), class_names)
  return class_names


def get_prompt_templates(prompt_templates_name,
                         *,
                         canonicalize=True):
  """Returns prompt templates."""
  prompts_templates = _PROMPT_TEMPLATES[prompt_templates_name]
  if canonicalize:
    prompts_templates = [
        canonicalize_text(name, keep_punctuation_exact_string="{}")
        for name in prompts_templates
    ]
  logging.info("Using %d prompts_templates: %s", len(prompts_templates),
               prompts_templates)
  return prompts_templates


def canonicalize_text(text, *, keep_punctuation_exact_string=None):
  """Returns canonicalized `text` (lowercase and puncuation removed).

  Args:
    text: string to be canonicalized.
    keep_punctuation_exact_string: If provided, then this exact string kept.
      For example providing '{}' will keep any occurrences of '{}' (but will
      still remove '{' and '}' that appear separately).
  """
  text = text.replace("_", " ")
  if keep_punctuation_exact_string:
    text = keep_punctuation_exact_string.join(
        part.translate(str.maketrans("", "", string.punctuation))
        for part in text.split(keep_punctuation_exact_string))
  else:
    text = text.translate(str.maketrans("", "", string.punctuation))
  text = text.lower()
  text = re.sub(r"\s+", " ", text)
  return text.strip()
