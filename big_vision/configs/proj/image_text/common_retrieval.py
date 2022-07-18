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

"""Common retrieval configuration."""

import ml_collections


def get_coco(
    *,
    pp_img='resize(224)|value_range(-1, 1)',
    pp_txt='tokenize(max_len=16, inkey="texts", eos="sticky", pad_value=1)',
    prefix='z/retr/coco_',
    log_steps):
  """Returns config for mscoco retrieval zero-shot.

  Args:
    pp_img: Pre-processing string for "image" feature.
    pp_txt: Pre-processing string for texts (expected to tokenize "texts" to
      "labels").
    prefix: Prefix to use for metrics.
    log_steps: How often the evaluators should be run.

  Returns:
    `ConfigDict` that can be used as a retrieval evaluator configuration.
  """
  return ml_collections.ConfigDict({
      'type': 'proj.image_text.retrieval',
      'log_steps': log_steps,
      'pp_txt': pp_txt,
      'pp_img': pp_img,
      'prefix': prefix,
      'dataset': 'coco_captions',
      'txt_name': ('captions', 'text'),
  })
