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

"""Common config code for LiT models."""

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
