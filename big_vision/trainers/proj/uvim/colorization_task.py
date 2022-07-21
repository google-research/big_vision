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

"""Inputs, outputs and losses for colorization task."""
import einops
import jax.numpy as jnp
import numpy as np

ONE_HOT_AXIS = -2


def input_pp(batch, config):
  """Make inputs for colorization task."""
  if "labels" not in batch:
    # During predict of phase2 there is no 'labels' field.
    x = None
  else:
    hp, wp = config.model.patch_size
    x = {
        "color": batch["labels"],
    }
    # Convert labels from (B, H, W) to (B, num_patches, C, patch_size)
    x["color"] = einops.rearrange(
        x["color"], "b (hn hp) (wn wp) c -> b (hn wn) c (hp wp)", hp=hp, wp=wp)
  ctx = batch.get("image_ctx", batch.get("image", None))
  return {"ctx": ctx, "x": x}


def loss_fn(logits, batch, config):
  """Compute loss for colorization task."""
  labels = input_pp(batch, config)["x"]
  error = logits["color"] - labels["color"]
  loss = jnp.square(error)
  return loss, {"loss_color": loss}


def predict_outputs(logits, config):
  """Make outputs for colorization task."""
  # Map logits to (height, width, channels).
  hp, wp = config.model.patch_size
  hn, wn = np.array(config.model.input_size) // np.array((hp, wp))
  assert ONE_HOT_AXIS == -2, "Rearrange below depends on this."
  output = einops.rearrange(
      logits["color"],
      "b (hn wn) c (hp wp) -> b (hn hp) (wn wp) c",
      hn=hn,
      wn=wn,
      hp=hp,
      wp=wp)
  output = jnp.clip(output, -1., 1.)
  return {"color": output}
