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

"""Scoring classifier.

This one is based on a generative perspective for image classification.
Here we input the image as well as all the tokenized labels to compute their
perplexity and select the one with minimum loss as the prediction.
"""
import functools
from big_vision.datasets.imagenet import class_names as imagenet_class_names
from big_vision.evaluators import mean
from big_vision.pp import builder as pp_builder
import jax.numpy as jnp
import numpy as np

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


CLASS_NAMES = {
    "imagenet2012": imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES,
}


# As a separate function to cache result across instances.
@functools.lru_cache(maxsize=None)
def get_classes(dataset_name, pp_txt):
  """Load the class label strings and tokenize them using pp_txt."""
  pp_fn = pp_builder.get_preprocess_fn(pp_txt, log_data=False)
  return np.array([pp_fn({"label": name})["labels"]
                   for name in CLASS_NAMES[dataset_name]])


def scoring(predict_fn, tokenized_labels):

  def _scoring_fn(train_state, batch, *a, **kw):
    batch = {"_label_tokens": tokenized_labels, **batch}
    scores = predict_fn(train_state, batch, *a, **kw)
    predictions = jnp.argmax(scores, axis=-1)
    return {"prec@1": predictions == batch["label"]}

  return _scoring_fn


class Evaluator(mean.Evaluator):
  """Evaluator for classification accuracy based on scoring all classes."""

  def __init__(self, predict_fn, data, pp_fn, pp_txt, *a, **kw):
    cls_tokens = get_classes(data["name"], pp_txt)
    super().__init__(scoring(predict_fn, cls_tokens), data, pp_fn, *a, **kw)
