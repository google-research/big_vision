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

"""Most common few-shot eval configuration."""

import ml_collections as mlc


def get_fewshot_lsr(target_resolution=224, resize_resolution=256,
                    runlocal=False):
  """Returns a standard-ish fewshot eval configuration."""
  config = mlc.ConfigDict()
  config.type = 'fewshot_lsr'
  config.representation_layer = 'pre_logits'
  config.log_steps = 25_000
  config.datasets = {
      'caltech': ('caltech101', 'train', 'test'),  # copybara:srtip
      'cars': ('cars196:2.1.0', 'train', 'test'),
      'cifar100': ('cifar100', 'train', 'test'),
      'dtd': ('dtd', 'train', 'test'),
      # The first 65000 ImageNet samples have at least 30 shots per any class.
      # Commented out by default because needs manual download.
      # 'imagenet': ('imagenet2012', 'train[:65000]', 'validation'),
      'pets': ('oxford_iiit_pet', 'train', 'test'),
      'uc_merced': ('uc_merced', 'train[:1000]', 'train[1000:]'),
  } if not runlocal else {
      'pets': ('oxford_iiit_pet', 'train', 'test'),
  }
  config.pp_train = f'decode|resize({resize_resolution})|central_crop({target_resolution})|value_range(-1,1)|keep("image", "label")'
  config.pp_eval = f'decode|resize({resize_resolution})|central_crop({target_resolution})|value_range(-1,1)|keep("image", "label")'
  config.shots = [1, 5, 10, 25]
  config.l2_reg = 2.0 ** 10
  config.num_seeds = 3
  config.display_first = [('imagenet', 10)] if not runlocal else [('pets', 10)]
  config.prefix = ''  # No prefix as we do already prefix with a/ z/ and zz/

  return config
