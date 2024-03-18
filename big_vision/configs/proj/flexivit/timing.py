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

# pylint: disable=line-too-long,missing-function-docstring
r"""A config to run timing for FlexiViT (only inference, no I/O etc.).

big_vision.tools.eval_only \
  --config big_vision/configs/proj/flexivit/timing.py \
  --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
  --config.total_epochs 90
"""

from ml_collections import ConfigDict


def get_config():
  c = ConfigDict()

  shape = (240, 240, 3)
  c.batch_size = 8  # swept
  c.init_shapes = [(1, *shape)]
  c.representation_layer = 'pre_logits'

  # Creating complete model using all params, the sweep will go over variants.
  c.model_name = 'xp.flexivit.vit'
  c.model = dict(
      variant='B',
      pool_type='tok',
      patch_size=(10, 10),  # Like deit@384
      seqhw=(24, 24),
  )
  c.num_classes = 0

  c.evals = {}
  c.evals.timing = dict(
      type='timing',
      input_shapes=[shape],
      timing=True,
      pred_kw=dict(outputs=('pre_logits',)),
  )

  return c