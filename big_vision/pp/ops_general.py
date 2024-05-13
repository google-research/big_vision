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

"""Generic tensor preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of (TF) tensors. The functors output a modified
dictionary.
"""

import collections

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
    with tf.io.gfile.GFile(mapping, "rb") as f:
      keys = np.array(np.load(f, allow_pickle=False)[npzkey])
    vals = np.arange(len(keys))

  # Otherwise, we simply use the file as a text file, with either of:
  # - a string per line, mapped to its line-number
  # - a pair, separated by `sep` per line, first value being the string, second
  #   value being the integer that the string is mapped to.
  else:
    with tf.io.gfile.GFile(mapping, "r") as f:
      buf = f.read()
    if sep is None:  # values are the line numbers
      keys = buf.splitlines()
      vals = np.arange(len(keys))
    else:  # each line is key<sep>val, also make val int
      keys, vals = zip(*[l.split(sep) for l in buf.splitlines()])
      vals = [int(v) for v in vals]

  def _do_the_mapping(needle):
    """Map string to number."""
    with tf.init_scope():  # (Originally added for performance reasons.)
      table = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(keys, vals), -1)
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
    labels = tf.cast(labels, tf.int64)  # both scatter and one_hot expect this
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
    data[outkey] = jax.tree.map(lambda x: x, data[inkey])
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
def get_pad_to_shape(shape, pad_value=0, where="after"):
  """Pads tensor to specified `shape`."""

  def _pads(cur, tgt):
    if tgt is None:
      return [0, 0]
    diff = tgt - cur
    return {
        "before": [diff, 0],
        "after": [0, diff],
        "both": [diff // 2, diff - diff // 2],
    }[where]

  def _pad_to_shape(x):
    assert len(x.shape.as_list()) == len(shape)
    paddings = [_pads(tgt=shape[i], cur=tf.shape(x)[i])
                for i in range(len(shape))]
    constant_value = tf.constant(pad_value, x.dtype)
    ret = tf.pad(x, paddings, constant_values=constant_value)
    ret.set_shape(shape)
    return ret

  return _pad_to_shape


@Registry.register("preprocess_ops.flatten")
def get_flatten():
  """Flattens the keys of data with separator '/'."""

  def flatten(data):
    flat, _ = bv_utils.tree_flatten_with_names(data)
    return dict(flat)

  return flatten


@Registry.register("preprocess_ops.reshape")
@utils.InKeyOutKey()
def get_reshape(new_shape):
  """Reshapes tensor to a given new shape.

  Args:
    new_shape: new shape for the tensor.

  Returns:
    A function for reshaping a tensor.

  """

  def _reshape(tensor):
    """Reshapes a tensor to a given shape."""
    dtype = tensor.dtype
    tensor = tf.reshape(tensor, new_shape)
    return tf.cast(tensor, dtype)

  return _reshape


@Registry.register("preprocess_ops.setdefault")
def get_setdefault(key, value):
  """If `key` is an empty tensor or missing, set it to `value`."""
  def _setdefault(data):
    x = data.get(key, tf.constant(value))
    v = tf.constant(value, dtype=x.dtype)
    v = tf.broadcast_to(v, [s or 1 for s in x.shape])
    data[key] = tf.cond(tf.size(x) > 0, lambda: x, lambda: v)
    return data
  return _setdefault


@Registry.register("preprocess_ops.choice")
def get_choice(n="single", key=None, fewer_ok=False, inkey=None, outkey=None):
  """Chooses the same `n` random entries of all `keys`.

  Args:
    n: how many entries to randomly sample (without repeat). Possible values:
      - int: that many entries (or fewer if there's fewer, see `fewer_ok`.)
      - "single": The string "single" only chooses one and drop the leading dim.
      - [min, max]: A pair means randomly take between min/max examples (incl.).
    key: str or list of str: See Note.
    fewer_ok: whether to fail when there's fewer than `n` elements to choose
              from (and hence set static shape to `n`), or whether to allow it.
              (and hence have unknown static shape).
    inkey: str or list of str: See Note.
    outkey: str or list of str: See Note.

  Note:
    If key/inkey/outkey is a list, then the same random entries are chosen for
    all of the keys. Other than that, they function the same as InKeyOutKey.

    The outkey can also contain the placeholder `{key}` that'll be .

  Examples:
    choice(key="alt_text/text")
    choice(n=128, key=["patches", "positions"])
    choice(inkey=["questions_i18n", "answers_i18n"], outkey=["q", "a"])

  Returns:
    The pp op.
  """

  # Normalize keys:
  inkeys = utils.maybe_repeat(inkey or key, 1)
  outkeys = utils.maybe_repeat(outkey or key, 1)
  outkeys = [ok.format(key=ik) for ok, ik in zip(outkeys, inkeys)]

  # Let's DRY on this condition and give it a name.
  is_varlen = isinstance(n, (list, tuple))
  min_n = n[0] if is_varlen else 1 if n == "single" else n

  def _choice(data):
    nitems = tf.shape(data[inkeys[0]])[0]

    # Sanity check that all keys have same leading dimension, and that is at
    # least as large as the minimum requested output.
    lengths = [tf.shape(data[k])[0] for k in inkeys]
    checks = [tf.debugging.assert_equal(l, nitems) for l in lengths]
    if not fewer_ok:  # Since we check for all-same, a single suffices here.
      checks.append(tf.debugging.assert_greater_equal(nitems, min_n))
    with tf.control_dependencies(checks):
      nitems = tf.identity(nitems)

    if n == "single":
      index = tf.random.uniform([], 0, nitems, dtype=tf.int32)
    else:
      # Subsample by shuffling and taking first n, but...
      indices = tf.random.shuffle(tf.range(nitems))
      end = n
      if is_varlen:
        end = tf.random.uniform([], n[0], n[1] + 1, dtype=tf.int32)
      # ...keep the order while subsampling (it might have a meaning, eg boxes)
      indices = tf.sort(indices[:end])

    for ik, ok in zip(inkeys, outkeys):
      if n == "single":
        result = data[ik][index]
      else:
        result = tf.gather(data[ik], indices, axis=0)
        if not is_varlen:  # Give static shape when we can.
          result = tf.ensure_shape(result, [n] + [None] * (result.ndim - 1))
      data[ok] = result

    return data
  return _choice


def _shuffled_index(count, nitems, seed):
  """Returns index from a shuffled sequence (items only repeat after epoch)."""
  nitems = tf.cast(nitems, count.dtype)
  item_epoch, item_offset = (count // nitems, count % nitems)
  shuffled_indices = tf.random.experimental.stateless_shuffle(
      tf.range(nitems), seed=tf.random.fold_in(seed, item_epoch))
  return shuffled_indices[item_offset]


@Registry.register("preprocess_ops.choice_no_replacement")
def get_choice_no_replacement(key=None, inkey=None, outkey=None):
  """Chooses the same random (no replacement) entry of all `keys`.

  Note: Consider using this for iterating over small datasets with a small
  number of epochs. It differs from `choice(n='single')` in that if an example,
  as identified by its `_id` field, is seen N times then it will cycled through
  all the inkeys values before repeating them. Additionally each repetition uses
  a different order.

  Caveats: requires dataset to provide a _id field and uses host RAM to keep a
  counter how often each id is seen. It is also not robust to preemptions.

  Args:
    key: str or list of str: See Note.
    inkey: str or list of str: See Note.
    outkey: str or list of str: See Note.

  Note:
    If key/inkey/outkey is a list, then the same random entries are chosen for
    all of the keys. Other than that, they function the same as InKeyOutKey.

    The outkey can also contain the placeholder `{key}` that'll be replaced
    by the inkey name.

  Examples:
    choice(key="alt_text/text")
    choice(key=["patches", "positions"])
    choice(inkey=["questions_i18n", "answers_i18n"], outkey=["q", "a"])

  Returns:
    The pp op.
  """
  # Normalize keys:
  inkeys = utils.maybe_repeat(inkey or key, 1)
  outkeys = utils.maybe_repeat(outkey or key, 1)
  outkeys = [ok.format(key=ik) for ok, ik in zip(outkeys, inkeys)]

  # TODO: Ideally the data pipeline should provide us with an epoch
  # counter. For now count how often we see a given example id and don't worry
  # on memory consumption. Counter returns 0 the first time an example is seen.
  counter = collections.defaultdict(lambda: -1)
  def _seen_count(example_id):
    example_id = example_id.item()
    counter[example_id] += 1
    return counter[example_id]

  # We need a seed to deterministically decide on a shuffled sequence and use
  # the number of times an example was seen to iterate through it. The seed
  # should be different for every instance of a create preprocessing function
  # but it has to be fixed for each instance.
  seed = tf.random.uniform(
      [2], minval=tf.int32.min, maxval=tf.int32.max, dtype=tf.int32)

  def _choice(data):
    nitems = tf.shape(data[inkeys[0]])[0]

    # Sanity check that all keys have same leading dimension.
    checks = [
        tf.debugging.assert_equal(tf.shape(data[k])[0], nitems)
        for k in inkeys
    ]
    with tf.control_dependencies(checks):
      nitems = tf.identity(nitems)

    # Using the seed, example id and the number of times an example was seen
    # pick an `index` such that items are only repeated after all items are seen
    # an equal number of times. E.g. it could return indexes from this sequence:
    #   [0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 2, 1, ...].
    count = tf.numpy_function(
        _seen_count, (data["_id"],), Tout=tf.int64, stateful=True)
    count = tf.cast(count, tf.int32)
    nitems = tf.cast(nitems, tf.int32)
    shuffle_epoch = count // nitems
    shuffle_offset = count % nitems

    example_seed = tf.random.fold_in(seed, data["_id"])
    shuffle_seed = tf.random.fold_in(example_seed, shuffle_epoch)
    shuffle = tf.random.experimental.stateless_shuffle(
        tf.range(nitems), seed=shuffle_seed)
    index = shuffle[shuffle_offset]

    # Select item[index] for all keys.
    for ik, ok in zip(inkeys, outkeys):
      data[ok] = data[ik][index]
    return data

  return _choice
