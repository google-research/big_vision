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

"""Evaluator for segmentation."""

import functools

import big_vision.evaluators.common as c
import big_vision.pp.tokenizer
import big_vision.utils as u
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image

from tensorflow.io import gfile


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


def _inrange(a, min_value, max_value):
  return (np.clip(a, min_value, max_value) == a).all()


def _area(y1, x1, y2, x2):
  return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


class Evaluator:
  """Evaluator for instance segmentation."""

  def __init__(self, predict_fn, tokenizer,
               model='oi', det_ious=(0.5, 0.75),
               *, devices, **kw):
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={'prefix', 'suffix', 'objects/mask', 'objects/bbox'},
        devices=devices, **kw)

    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token)
    tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.loc0 = np.array(tok.to_int('<loc0000>'))
    self.seg0 = np.array(tok.to_int('<seg000>'))
    # Verify tokenizer has `tokensets=("loc", "seg")`
    assert self.loc0.shape == (1,), self.loc0
    assert self.seg0.shape == (1,), self.seg0
    self.reconstruct_masks = get_reconstruct_masks(model)
    self.det_ious = det_ious

  def run(self, train_state):
    """Does one evaluation run, yields metrics."""
    ious = []  # NOTE: no point to split in s/m/l: all objects are L (>96px²)
    det_by_iou = {iou: [] for iou in self.det_ious}
    invalid = total = 0
    for _, batch in zip(range(self.steps), self.get_data_iter()):

      decoded = self.decode(train_state, batch)

      not_padding = u.get_local_slice_from_fsarray(batch['_mask'])
      decoded = u.get_local_slice_from_fsarray(decoded)[not_padding]

      # Note, gt masks are in full original image resolution.
      gt_masks = [gt[:, :, 0] > 0 for gt in batch['objects/mask'][not_padding]]
      gt_bbs = [gt for gt in batch['objects/bbox'][not_padding]]

      valid = []
      tokens = np.zeros([decoded.shape[0], 4 + 16], np.int32)
      for i, dec in enumerate(decoded):
        # TODO: b/andstein - do we need to optimize this loop?
        t = np.r_[dec[:4] - self.loc0, dec[4:4 + 16] - self.seg0]  # Ignore rest
        if (
            len(t) == 4 + 16  # Full prediction
            and _inrange(t[:4], 0, 1023)  # Valid box tokens
            and _inrange(t[4:], 0, 127)  # Valid seg tokens
            and t[2] > t[0] and t[3] > t[1]  # Valid box
        ):
          valid.append(True)
          tokens[i] = t
        else:
          valid.append(False)

      tocpu = lambda x: jax.device_put(x, jax.local_devices(backend='cpu')[0])
      seg_indices = np.array(tokens[:, 4:])
      mask64 = jax.device_get(self.reconstruct_masks(tocpu(seg_indices)))
      mask64 = mask64[..., 0]
      bbox = tokens[:, :4] / 1023  # Back to [0.0 ... 1.0]

      for v, m64, gtm, bb, gtbb in zip(valid, mask64, gt_masks, bbox, gt_bbs):
        # TODO: b/andstein - do we need to optimize this loop?
        total += 1
        h, w = gtm.shape  # gt is full/original image resolution mask.

        # First, compute detection iou, in [0.0 ... 1.0] coordinate space.
        y1, x1, y2, x2 = bb
        gty1, gtx1, gty2, gtx2 = gtbb
        ibb = max(y1, gty1), max(x1, gtx1), min(y2, gty2), min(x2, gtx2)
        box_iou = _area(*ibb) / (_area(*bb) + _area(*gtbb) - _area(*ibb))
        for iou_thresh in det_by_iou:
          det_by_iou[iou_thresh].append(iou_thresh <= box_iou)

        # Next, we convert to pixel coordinates and compute mask iou.
        gt_area = gtm.sum()
        y1, x1, y2, x2 = map(int, (y1 * h, x1 * w, y2 * h, x2 * w))

        # Avoid compute-intensive mask stuff for invalid preds:
        if not v or x2 <= x1 or y2 <= y1:  # Can still happen after int().
          iou = 0.0
          invalid += 1
        else:
          mi = np.asarray(PIL.Image.fromarray(m64).resize(
              [x2 - x1, y2 - y1], resample=PIL.Image.BILINEAR  # pytype: disable=module-attr
          ))  # Predicted mask in box-sized image.
          mi = mi > 0.0  # Mask decoder output in [-1.0 ... 1.0]
          iarea = (gtm[y1:y2, x1:x2] & mi).sum()  # Intersection pixels.
          iou = iarea / (gt_area + mi.sum() - iarea)
        ious.append(iou)

    # Done going over all batches, now collect results from all processes.
    sum_ious, num_ious, sum_dets, num_dets, num_invalid, num = c.process_sum([
        sum(ious), len(ious),
        {k: sum(v) for k, v in det_by_iou.items()},
        {k: len(v) for k, v in det_by_iou.items()},
        invalid, total
    ])

    yield 'miou', sum_ious / num_ious
    for k in sum_dets:
      yield f'boxacc/{k}', sum_dets[k] / num_dets[k]
    yield 'invalid', num_invalid
    yield 'total', num


_KNOWN_MODELS = {
    # Trained on open images.
    'oi': 'gs://big_vision/paligemma/vae-oid.npz',
}


def _get_params(checkpoint):
  """Converts PyTorch checkpoint to Flax params."""

  def transp(kernel):
    return np.transpose(kernel, (2, 3, 1, 0))

  def conv(name):
    return {
        'bias': checkpoint[name + '.bias'],
        'kernel': transp(checkpoint[name + '.weight']),
    }

  def resblock(name):
    return {
        'Conv_0': conv(name + '.0'),
        'Conv_1': conv(name + '.2'),
        'Conv_2': conv(name + '.4'),
    }

  return {
      '_embeddings': checkpoint['_vq_vae._embedding'],
      'Conv_0': conv('decoder.0'),
      'ResBlock_0': resblock('decoder.2.net'),
      'ResBlock_1': resblock('decoder.3.net'),
      'ConvTranspose_0': conv('decoder.4'),
      'ConvTranspose_1': conv('decoder.6'),
      'ConvTranspose_2': conv('decoder.8'),
      'ConvTranspose_3': conv('decoder.10'),
      'Conv_1': conv('decoder.12'),
  }


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
  batch_size, num_tokens = codebook_indices.shape
  assert num_tokens == 16, codebook_indices.shape
  unused_num_embeddings, embedding_dim = embeddings.shape

  encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
  encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
  return encodings


class ResBlock(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x):
    original_x = x
    x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
    x = nn.relu(x)
    x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
    x = nn.relu(x)
    x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
    return x + original_x


class Decoder(nn.Module):
  """Upscales quantized vectors to mask."""

  @nn.compact
  def __call__(self, x):
    num_res_blocks = 2
    dim = 128
    num_upsample_layers = 4

    x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
    x = nn.relu(x)

    for _ in range(num_res_blocks):
      x = ResBlock(features=dim)(x)

    for _ in range(num_upsample_layers):
      x = nn.ConvTranspose(
          features=dim,
          kernel_size=(4, 4),
          strides=(2, 2),
          padding=2,
          transpose_kernel=True,
      )(x)
      x = nn.relu(x)
      dim //= 2

    x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)

    return x


@functools.cache
def get_reconstruct_masks(model):
  """Reconstructs masks from codebook indices.

  Based on code from https://arxiv.org/abs/2301.02229

  Verified in
  https://colab.research.google.com/drive/1AOr0cokOpM6-N9Z5HmxoeGxGj6jS37Vl

  Args:
    model: Model to use for conversion.

  Returns:
    A function that expects indices shaped `[B, 16]` of dtype int32, each
    ranging from 0 to 127 (inclusive), and that returns a decoded masks sized
    `[B, 64, 64, 1]`, of dtype float32, in range [-1, 1].
  """
  def reconstruct_masks(codebook_indices):
    quantized = _quantized_values_from_codebook_indices(
        codebook_indices, params['_embeddings']
    )
    return Decoder().apply({'params': params}, quantized)

  with gfile.GFile(_KNOWN_MODELS.get(model, model), 'rb') as f:
    params = _get_params(dict(np.load(f)))

  return jax.jit(reconstruct_masks, backend='cpu')
