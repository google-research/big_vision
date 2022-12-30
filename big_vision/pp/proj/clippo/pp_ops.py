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

"""Preprocessing functions for CLIP with Pixels Only (CLIPPO)."""
from absl import logging
from  big_vision.pp import utils
from big_vision.pp.registry import Registry
import numpy as np
import tensorflow as tf


@Registry.register("preprocess_ops.render_unifont")
@utils.InKeyOutKey(indefault="texts", outdefault="image")
def get_pp_render_text(image_size: int, font_size: int = 16, max_chars=768,
                       background_brightness=127, text_brightness=0,
                       lower=True, monospace=False, spacing=1, min_width=4,
                       resize_method="area"):
  """Renders text as image, using binary Unifont.

  Largely based on Jeffrey Sorensen's text rendering implementation.

  Args:
    image_size: Width/height of output image.
    font_size: Font size to use. Recommended to leave at 16, as this requires
    no resizing, and is safe.
    max_chars: Maximum inpute characters to render, to make faster.
    background_brightness: (r, g, b) of background pixels.
    text_brightness: (r, g, b) of text pixels.
    lower: whether to lowercase.
    monospace: if False, text characters are horizontally trimmed according to
      `spacing` and `minwidth` args.
    spacing: # pixels between each letter.
    min_width: Minimum width of each letter. Useful to make sure e.g. spaces and
      full stops aren't collapsed to nothing.
    resize_method: resize method to use if fontsize != 16.

  Returns:
    Function which renders text as an image.
  """
  bit_embedding = np.zeros((0x200000, 32), dtype=np.uint8)
  colpattern = {64: range(32),
                32: sorted(tuple(range(0, 32, 4)) + tuple(range(2, 32, 4)))}

  unifont_path = "big_vision/pp/proj/clippo/unifont-9.0.06.hex"
  unifont_upper_path = "big_vision/pp/proj/clippo/unifont_upper-9.0.06.hex"

  with tf.io.gfile.GFile(unifont_path) as f:
    for line in f:
      row = int(line[0:4], 16)
      hexbits = line[5:-1]
      bit_embedding[row, colpattern[len(hexbits)]] = bytearray.fromhex(hexbits)

  with tf.io.gfile.GFile(unifont_upper_path) as f:
    for line in f:
      row = int(line[0:6], 16)
      hexbits = line[7:-1]
      bit_embedding[row, colpattern[len(hexbits)]] = bytearray.fromhex(hexbits)

  params = tf.constant(bit_embedding, dtype=tf.uint8)

  def trim_letter(letter):
    """Remove white space based on the letter size."""
    v = tf.reduce_max(letter, axis=0)
    has_pixels = tf.reshape(tf.where(v), (-1,), name="RS5")
    no_pixels = tf.equal(tf.reduce_max(v), 0)
    first = tf.cond(no_pixels, lambda: tf.constant(0, tf.int64),
                    lambda: has_pixels[0])
    last = tf.cond(no_pixels, lambda: tf.constant(0, tf.int64),
                   lambda: has_pixels[-1])

    first = tf.maximum(first - spacing, 0)
    last = tf.maximum(last + spacing, first + min_width)
    return tf.RaggedTensor.from_tensor(tf.transpose(letter[:, first:last]))

  def to_image(rendered, width, height=None):
    """Makes a nice square image from a long string of rendered charcaters."""
    height = height or width
    max_letter_width = tf.reduce_max(rendered.row_lengths(1))
    row_lengths = tf.cast(tf.cumsum(rendered.row_lengths(1)), tf.float32)
    div = tf.cast(width - max_letter_width, tf.float32)  # For rounding errors.
    row_idx = tf.cast(tf.floor(row_lengths / div), tf.int64)
    row_idx = tf.RaggedTensor.from_value_rowids(tf.range(tf.shape(rendered)[0]),
                                                row_idx)
    trimmed = tf.gather(rendered, row_idx, axis=0)
    trimmed = trimmed.merge_dims(1, 2)
    trimmed = trimmed.to_tensor(default_value=0)
    trimmed = tf.transpose(trimmed, (0, 2, 1))
    trimmed = tf.reshape(trimmed, (-1, tf.shape(trimmed)[-1]), name="RS4")
    trimmed = trimmed[:height]

    wpad = width - tf.shape(trimmed)[1]
    hpad = height - tf.shape(trimmed)[0]
    padded = tf.pad(trimmed, [[0, hpad], [0, wpad]])
    tf.assert_equal(tf.shape(padded), tf.constant((height, width)))
    return tf.ensure_shape(padded, (width, height))

  def render(text):
    if lower:
      text = tf.strings.lower(text)
    text = tf.reshape(text, (-1,))[0]
    ids = tf.strings.unicode_decode(text, "UTF-8")
    if max_chars:
      ids = ids[:max_chars]
    embed = tf.nn.embedding_lookup(params, ids)  # Get the letters
    # Each letter is 32 uint8s, but we want binary 16x16 grid.
    # The following does that in a rather hard to parse way.
    vertical = tf.reshape(embed, [1, -1])
    repl = tf.reshape(tf.transpose(tf.tile(vertical, multiples=[8, 1])), [-1])
    ones = tf.ones_like(repl)
    index = tf.cumsum(ones, exclusive=True)
    sevens = tf.cast(tf.fill(tf.shape(repl), 7), tf.uint8)
    moded = tf.bitwise.bitwise_and(index, sevens)
    shifted = tf.bitwise.right_shift(repl,
                                     tf.bitwise.bitwise_xor(moded, sevens))
    anded = tf.bitwise.bitwise_and(shifted, ones)
    # And finally, letters; binary, 0 = background, 1 = letter.
    letters = tf.reshape(anded, [tf.shape(ids)[0], 16, 16])

    if font_size != 16:
      logging.warning("The unifont text rendering function is highly optimized "
                      "for font size 16; using font size %i might lead to "
                      "suboptimal rendering and might degrade performance.",
                      font_size)
      letters = tf.image.resize(letters[..., None], (font_size, font_size),
                                method=resize_method, antialias=True)
      letters = tf.squeeze(letters, axis=-1)

    if monospace:
      letters = tf.RaggedTensor.from_tensor(tf.transpose(letters, (0, 2, 1)))
    else:
      letters = tf.RaggedTensor.from_tensor(letters)
      signature = tf.RaggedTensorSpec(shape=(None, font_size), ragged_rank=1,
                                      dtype=letters.dtype)
      letters = tf.map_fn(trim_letter, letters, fn_output_signature=signature)

    img = to_image(letters, image_size)[..., None]    # A nice square image.
    img *= (text_brightness - background_brightness)  # Rescale value range.
    img += background_brightness

    return tf.image.grayscale_to_rgb(tf.cast(img, tf.uint8))

  return render
