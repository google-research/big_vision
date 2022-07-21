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

"""Inputs, outputs and losses for panoptic task."""
import big_vision.utils as u
import einops
import jax
import jax.numpy as jnp
import numpy as np

ONE_HOT_AXIS = -2


def input_pp(batch, config):
  """Make inputs for panoptic segmentation task."""
  if "labels" not in batch:
    # During predict of phase2 there is no 'labels' field.
    x = None
  else:
    hp, wp = config.model.patch_size
    x = {
        "semantics": batch["labels"][..., 0],
        "instances": batch["labels"][..., 1],
    }
    # Convert labels from (B, H, W) to (B, num_patches, num_classes, patch_size)
    for key in ["semantics", "instances"]:
      x[key] = jax.nn.one_hot(
          einops.rearrange(
              x[key], "b (hn hp) (wn wp) -> b (hn wn) (hp wp)", hp=hp, wp=wp),
          num_classes=config.model.inputs[key][ONE_HOT_AXIS], axis=ONE_HOT_AXIS)
  ctx = batch.get("image_ctx", batch.get("image", None))
  return {"ctx": ctx, "x": x}


def loss_fn(logits, batch, config):
  """Compute loss for panoptic task."""
  labels = input_pp(batch, config)["x"]
  losses = {}
  for key in ["semantics", "instances"]:
    losses[f"loss_{key}"] = u.softmax_xent(
        logits=logits[key], labels=labels[key], reduction=False,
        axis=ONE_HOT_AXIS)
  return sum(losses.values()), losses


def predict_outputs(logits, config, min_fraction=0.0):
  """Make outputs for panoptic segmentation task."""
  # Map logits to (height, width, channels).
  hp, wp = config.model.patch_size
  hn, wn = np.array(config.model.input_size) // np.array((hp, wp))
  outputs = {}
  for key in ["semantics", "instances"]:
    assert ONE_HOT_AXIS == -2, "Rearrange below depends on this."
    outputs[key] = einops.rearrange(
        logits[key],
        "b (hn wn) c (hp wp) -> b (hn hp) (wn wp) c",
        hn=hn, wn=wn, hp=hp, wp=wp)
  return panoptic_predictions_from_logits(
      **outputs, min_fraction=min_fraction)


def panoptic_predictions_from_logits(semantics, instances, min_fraction=0.0):
  """Make panoptic prediction from logits."""
  ins = jnp.argmax(instances, axis=-1)
  # Note: Make sure each instance has all pixels annotated with same label.
  # Otherwise they are further split into more instances and greatly affect
  # the number of unmatched predicted segments (FP) and RQ.
  masks = jax.nn.one_hot(ins, instances.shape[-1], dtype=jnp.int32)
  label = jnp.argmax(jnp.einsum("bhwk,bhwn->bnk", semantics, masks), axis=-1)
  sem = jnp.einsum("bhwn,bn->bhw", masks, label)
  out = jnp.stack([sem, ins], axis=-1)
  # Filter out small objects
  fraction = jnp.sum(masks, axis=(1, 2), keepdims=True)/np.prod(ins.shape[1:3])
  mask_big = (fraction > min_fraction).astype("int32")
  mask_big_spatial = jnp.sum(masks * mask_big, axis=-1, keepdims=True) > 0
  return out * mask_big_spatial.astype("int32")
