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

"""Inputs, outputs and losses for depth prediction task."""
import big_vision.utils as u
import einops
import jax
import jax.numpy as jnp
import numpy as np


ONE_HOT_AXIS = -2


def input_pp(batch, config):
  """Makes inputs for depth prediction task."""
  if "labels" not in batch:
    x = None
  else:
    hp, wp = config.model.patch_size
    depth = batch["labels"][..., 0]

    # Discretize to [0, ..., bins - 1].
    nbins = config.model.inputs.depth[ONE_HOT_AXIS]
    mind = config.min_depth
    maxd = config.max_depth
    depth = (depth - mind) / (maxd - mind)
    depth *= nbins
    depth = jnp.floor(depth).astype(jnp.int32)
    depth = jnp.minimum(depth, nbins - 1)
    depth = jnp.maximum(depth, 0)

    # Converts labels from (B, H, W, c) to (B, num_patches, c, patch_size).
    depth = jax.nn.one_hot(
        einops.rearrange(
            depth, "b (hn hp) (wn wp) -> b (hn wn) (hp wp)", hp=hp, wp=wp),
        num_classes=config.model.inputs.depth[ONE_HOT_AXIS],
        axis=ONE_HOT_AXIS)
    x = {"depth": depth}
  ctx = batch.get("image_ctx", batch.get("image", None))
  return {"ctx": ctx, "x": x}


def loss_fn(predictions, batch, config):
  """Computes loss for depth prediction task."""
  labels = input_pp(batch, config)["x"]
  losses = {}
  loss = u.softmax_xent(
      logits=predictions["depth"], labels=labels["depth"], reduction=False,
      axis=ONE_HOT_AXIS)
  # Do not train on the closest class; usually regions of the image with
  # depth==0, which is the default for regions with no depth signal.
  # TODO: Encode depth==0 as class==-1.
  mask = jnp.argmax(labels["depth"], ONE_HOT_AXIS) != 0
  loss = loss * mask
  losses["loss_depth"] = loss
  return sum(losses.values()), losses


def predict_outputs(predictions, config):
  """Makes outputs for depth predictin tasks."""
  # Maps predictions to (height, width, channels).
  hp, wp = config.model.patch_size
  hn, wn = np.array(config.model.input_size) // np.array((hp, wp))
  depth = einops.rearrange(
      predictions["depth"],
      "b (hn wn) c (hp wp) -> b (hn hp) (wn wp) c",
      hn=hn, wn=wn, hp=hp, wp=wp)

  depth = jnp.argmax(depth, axis=-1)  # [B, H, W]

  # Revert discretization.
  nbins = config.model.inputs.depth[ONE_HOT_AXIS]
  mind = config.min_depth
  maxd = config.max_depth
  depth = depth.astype(jnp.float32) + 0.5  # Undoes floor in expectation.
  depth /= nbins
  depth = depth * (maxd - mind) + mind

  return {"depth": depth}
