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

"""Generic tensor preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of (TF) tensors. The functors output a modified
dictionary.
"""

from big_vision.pp import utils
from big_vision.pp.registry import Registry
import big_vision.utils as bv_utils
import jax
import numpy as np
import tensorflow as tf


@Registry.register("preprocess_ops.value_range")
@utils.InKeyOutKey()
def get_value_range(vmin=-1, vmax=1, in_min=0, in_max=255.0, clip_values=False):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.

  Returns:
    A function to rescale the values.
  """

  def _value_range(image):
    """Scales values in given range."""
    in_min_t = tf.constant(in_min, tf.float32)
    in_max_t = tf.constant(in_max, tf.float32)
    image = tf.cast(image, tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = vmin + image * (vmax - vmin)
    if clip_values:
      image = tf.clip_by_value(image, vmin, vmax)
    return image

  return _value_range


@Registry.register("preprocess_ops.lookup")
@utils.InKeyOutKey()
def get_lookup(mapping, npzkey="fnames", sep=None):
  """Map string to number."""

  # For NumPy files, we use the `npzkey` array in that file as the list of
  # strings which are mapped to their index in that array.
  # This is especially useful when other data (eg precomputed predictions)
  # goes along with this mapping, to have everything in one place (the npz).
  if mapping.endswith(".npz"):
    with tf.gfile.GFile(mapping, "rb") as f:
      keys = np.array(np.load(f, allow_pickle=False)[npzkey])
    vals = np.arange(len(keys))

  # Otherwise, we simply use the file as a text file, with either of:
  # - a string per line, mapped to its line-number
  # - a pair, separated by `sep` per line, first value being the string, second
  #   value being the integer that the string is mapped to.
  else:
    with tf.gfile.GFile(mapping, "rt") as f:
      buf = f.read()
    if sep is None:  # values are the line numbers
      keys = buf.splitlines()
      vals = np.arange(len(keys))
    else:  # each line is key<sep>val, also make val int
      keys, vals = zip(*[l.split(sep) for l in buf.splitlines()])
      vals = [int(v) for v in vals]

  def _do_the_mapping(needle):
    """Map string to number."""

    with tf.init_scope():
      init = tf.lookup.KeyValueTensorInitializer(
          tf.constant(keys), tf.constant(vals))
      table = tf.contrib.lookup.HashTable(init, -1)

    return table.lookup(needle)

  return _do_the_mapping


@Registry.register("preprocess_ops.onehot")
def get_onehot(depth,
               key="labels",
               key_result=None,
               multi=True,
               on=1.0,
               off=0.0):
  """One-hot encodes the input.

  Args:
    depth: Length of the one-hot vector (how many classes).
    key: Key of the data to be one-hot encoded.
    key_result: Key under which to store the result (same as `key` if None).
    multi: If there are multiple labels, whether to merge them into the same
      "multi-hot" vector (True) or keep them as an extra dimension (False).
    on: Value to fill in for the positive label (default: 1).
    off: Value to fill in for negative labels (default: 0).

  Returns:
    Data dictionary.
  """

  def _onehot(data):
    # When there's more than one label, this is significantly more efficient
    # than using tf.one_hot followed by tf.reduce_max; we tested.
    labels = data[key]
    if labels.shape.rank > 0 and multi:
      x = tf.scatter_nd(labels[:, None], tf.ones(tf.shape(labels)[0]), (depth,))
      x = tf.clip_by_value(x, 0, 1) * (on - off) + off
    else:
      x = tf.one_hot(labels, depth, on_value=on, off_value=off)
    data[key_result or key] = x
    return data

  return _onehot


@Registry.register("preprocess_ops.keep")
def get_keep(*keys):
  """Keeps only the given keys."""

  def _keep(data):
    return {k: v for k, v in data.items() if k in keys}

  return _keep


@Registry.register("preprocess_ops.drop")
def get_drop(*keys):
  """Drops the given keys."""

  def _drop(data):
    return {k: v for k, v in data.items() if k not in keys}

  return _drop


@Registry.register("preprocess_ops.copy")
def get_copy(inkey, outkey):
  """Copies value of `inkey` into `outkey`."""

  def _copy(data):
    # A "semi-deep" copy. deepcopy doesn't work when tf tensors are part of the
    # game. What we want, is to only copy the python structure (dicts, lists)
    # and keep tensors as they are, since we never modify them in-place anyways.
    # The following achieves exactly that.
    data[outkey] = jax.tree_map(lambda x: x, data[inkey])
    return data

  return _copy


@Registry.register("preprocess_ops.squeeze_last_dim")
@utils.InKeyOutKey()
def get_squeeze_last_dim():
  def _squeeze_last_dim(x):
    return tf.squeeze(x, axis=-1)
  return _squeeze_last_dim


@Registry.register("preprocess_ops.concat")
def get_concat(inkeys, outkey=None, axis=-1):
  """Concatenates elements along some axis."""

  def _concat(data):
    data[outkey or inkeys[0]] = tf.concat([data[k] for k in inkeys], axis)
    return data

  return _concat


@Registry.register("preprocess_ops.rag_tensor")
@utils.InKeyOutKey()
def get_rag_tensor():
  """Converts the specified feature to ragged tensor."""

  def rag_tensor(raw_tensor):
    # Note: Add one more dimension as `from_tensor` requires at least rank 2.
    return tf.RaggedTensor.from_tensor(raw_tensor[None])

  return rag_tensor


@Registry.register("preprocess_ops.pad_to_shape")
@utils.InKeyOutKey()
def get_pad_to_shape(shape, pad_value=0):
  def _pad_to_shape(x):
    assert len(x.shape.as_list()) == len(shape)
    paddings = [[0, shape[i] - tf.shape(x)[i]] for i in range(len(shape))]
    return tf.pad(x, paddings, constant_values=tf.constant(pad_value, x.dtype))

  return _pad_to_shape


@Registry.register("preprocess_ops.flatten")
def get_flatten():
  """Flattens the keys of data with separator '/'."""

  def flatten(data):
    flat, _ = bv_utils.tree_flatten_with_names(data)
    return dict(flat)

  return flatten
