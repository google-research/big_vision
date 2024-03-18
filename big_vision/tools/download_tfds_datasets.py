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

"""Download and prepare TFDS datasets for the big_vision codebase.

This python script covers cifar10, cifar100, oxford_iiit_pet
and oxford_flowers10.

If you want to integrate other public or custom datasets, please follow:
https://www.tensorflow.org/datasets/catalog/overview
"""

from absl import app
import tensorflow_datasets as tfds


def main(argv):
  if len(argv) > 1 and "download_tfds_datasets.py" in argv[0]:
    datasets = argv[1:]
  else:
    datasets = [
        "cifar10",
        "cifar100",
        "oxford_iiit_pet",
        "oxford_flowers102",
        "imagenet_v2",
    ]
  for d in datasets:
    tfds.load(name=d, download=True)


if __name__ == "__main__":
  app.run(main)
