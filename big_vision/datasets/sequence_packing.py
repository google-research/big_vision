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

"""Packed Sequence Op."""

# Forked from
# https://github.com/google/maxtext/blob/main/MaxText/sequence_packing.py.


from typing import Dict, Optional, List, Union

from flax import traverse_util
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLATTEN_SEPARATOR = "<|sep|>"


def pack_dataset(
    dataset: tf.data.Dataset,
    batch_size: int | None,
    key2length: Union[int, Dict[str, int]],
    keys: Optional[List[str | tuple[str, ...]]] = None) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.
 
  Wrap `tensorflow.grain` ops.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.
  Each example in the output dataset represents several examples in the
  input dataset.

  For each key in the input dataset, two additional keys are created:
  <key>_segment_ids: an int32 tensor identifying the parts
     representing the original example.
  <key>_positions: an int32 tensor identifying the position within the original
     example.

  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
             "inputs_seg": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
             "inputs_pos": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
            "targets_seg": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
            "targets_pos": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }
  0 represents padding in both the inputs and the outputs.
  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: A `tf.data.Dataset`.
    batch_size: Batch size of the packed dataset.
    key2length: An integer, or a dict from feature-key to integer.
    keys: A list of strings (e.g. ["inputs", "targets"]).

  Returns:
    A `tf.data.Dataset`.
  """
  raise ValueError("Not implemented in OSS yet.")
