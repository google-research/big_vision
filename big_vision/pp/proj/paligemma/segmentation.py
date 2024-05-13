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

"""Segmentation-related ops."""

import functools

from big_vision.pp import registry
import numpy as np
import tensorflow as tf

from tensorflow.io import gfile


_KNOWN_MODELS = {
    'oi': 'gs://big_vision/paligemma/vae-oid.npz',
}


@functools.cache
def get_checkpoint(model):
  with gfile.GFile(_KNOWN_MODELS.get(model, model), 'rb') as f:
    return dict(np.load(f))


@registry.Registry.register('preprocess_ops.refcoco_mask2str')
def get_refcoco_mask2str(model='oi'):
  """Returns op for tokenizing a mask."""

  seg_tokens = tf.constant(['<seg%03d>' % i for i in range(128)])
  loc_tokens = tf.constant(['<loc%04d>' % i for i in range(1024)])
  checkpoint = get_checkpoint(model)

  def refcoco_mask2str(data):

    mask = data['objects/mask']
    tf.ensure_shape(mask, [None, None, 3])  # requires choice()
    sentence = data['objects/refs/sentence']
    tf.ensure_shape(sentence, [])  # requires choice()
    bbox = data['objects/bbox']
    tf.ensure_shape(bbox, [4])  # requires choice()

    h = tf.cast(tf.shape(mask)[0], tf.float32)
    w = tf.cast(tf.shape(mask)[1], tf.float32)
    y1 = tf.cast(tf.round(h * bbox[0]), tf.int32)
    x1 = tf.cast(tf.round(w * bbox[1]), tf.int32)
    y2 = tf.cast(tf.round(h * bbox[2]), tf.int32)
    x2 = tf.cast(tf.round(w * bbox[3]), tf.int32)

    assert mask.dtype == tf.uint8, mask.dtype
    mask = tf.image.resize(
        mask[None, y1:y2, x1:x2, :1],
        [64, 64],
        method='bilinear',
        antialias=True,
    ) / 255.0

    mask_indices = encode_to_codebook_indices(checkpoint, mask)[0]
    mask_string = tf.strings.reduce_join(tf.gather(seg_tokens, mask_indices))

    binned_loc = tf.cast(tf.round(bbox * 1023), tf.int32)
    binned_loc = tf.clip_by_value(binned_loc, 0, 1023)
    loc_string = tf.strings.reduce_join(tf.gather(loc_tokens, binned_loc))

    data['prefix'] = sentence
    data['suffix'] = tf.strings.join([loc_string, mask_string])

    return data

  return refcoco_mask2str


# Based on https://arxiv.org/abs/2301.02229.

NUM_DOWNSAMPLE_LAYERS = 4
NUM_RES_BLOCKS = 2


def encode_to_codebook_indices(checkpoint, masks):
  """Encode a batch of binary segmentation masks into 16 tokens each.

  Based on code from https://arxiv.org/abs/2301.02229

  Args:
    checkpoint: model weights from PyTorch model.
    masks: Must be in range `[0..1]`, and of shape `[None, 64, 64, 1]`.

  Returns:
    A tensor of shape `[None, 16]` with elements in `range(128)`.
  """

  # We require that the input masks are already resized to 64x64.
  x = tf.ensure_shape(masks, [None, 64, 64, 1])
  x = _norm(x)

  for n in range(NUM_DOWNSAMPLE_LAYERS):
    x = _conv_tf(
        checkpoint, x, strides=2, padding='SAME', layer_name=f'encoder.{2*n}'
    )
    x = tf.nn.relu(x)

  for n in range(NUM_RES_BLOCKS):
    x = _resblock_tf(checkpoint, x, layer_name=f'encoder.{8+n}.net')

  x = _conv_tf(
      checkpoint, x, strides=1, padding='SAME', layer_name='encoder.10'
  )

  return _get_codebook_indices(checkpoint, x)


def _norm(x):
  return 2.0 * (x - 0.5)


def _conv_tf(checkpoint, x, strides, padding, layer_name):
  kernel = checkpoint[layer_name + '.weight']
  kernel = np.transpose(kernel, (2, 3, 1, 0))
  bias = checkpoint[layer_name + '.bias']
  return tf.nn.conv2d(x, kernel, strides=strides, padding=padding) + bias


def _resblock_tf(checkpoint, x, layer_name):
  """Apply a residual block of the mask encoder."""
  original_x = x
  x = _conv_tf(
      checkpoint, x, padding='SAME', strides=1, layer_name=layer_name + '.0'
  )
  x = tf.nn.relu(x)
  x = _conv_tf(
      checkpoint, x, padding='SAME', strides=1, layer_name=layer_name + '.2'
  )
  x = tf.nn.relu(x)
  x = _conv_tf(
      checkpoint, x, padding='SAME', strides=1, layer_name=layer_name + '.4'
  )
  return x + original_x


def _get_codebook_indices(checkpoint, encoder_output):
  embeddings = checkpoint['_vq_vae._embedding']
  flat_input = tf.reshape(encoder_output, [-1, embeddings.shape[1]])
  distances = (
      tf.reduce_sum(flat_input**2, axis=1, keepdims=True)
      + tf.reduce_sum(embeddings**2, axis=1)
      - 2 * tf.matmul(flat_input, embeddings.T)
  )
  indices = tf.argmin(distances, axis=1)
  return tf.reshape(indices, [-1, 16])
