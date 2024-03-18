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

"""RandAug depends on deprecated tfa.image package, now defunct."""
from big_vision.pp import autoaugment
from big_vision.pp import utils
from big_vision.pp.registry import Registry


@Registry.register("preprocess_ops.randaug")
@utils.InKeyOutKey()
def get_randaug(num_layers: int = 2, magnitude: int = 10):
  """Creates a function that applies RandAugment.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,

  Args:
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].

  Returns:
    a function that applies RandAugment.
  """

  def _randaug(image):
    return autoaugment.distort_image_with_randaugment(
        image, num_layers, magnitude)

  return _randaug
