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

"""Object detection reward from "Tuning computer vision models with task rewards" (https://arxiv.org/abs/2302.08242).

The `reward_fn` computes the reward for a batch of predictions and ground truth
annotations. When using it to optimize a model that outputs a prediction as a
sequence of tokens like [y0, x0, Y0, X0, class0, confidence0, y1, x1, Y1, ...]
the training loop may look like:

```
# Settings used in the paper.
config.max_level = 1000  # Coordinates are discretized into 1000 buckets.
config.max_conf = 2 # Two tokens are reserved to represent confidence.
config.num_cls = 80 # Number of classes in COCO.
config.nms_w = 0.3  # Weight for duplicate instances.
config.cls_smooth = 0.05  # Adjust the classes weights based on their frequency.
config.reward_thr = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
config.correct_thr = 0.5  # Learn the IoU when matching with threshold=0.5.
config.conf_w = 0.3  # Weight for the confidence loss.


# 1) Sample N outputs for each input and compute rewards, use one sample to
# optimize and others to compute a reward baseline.
sample_seqs = sample_fn(params, images, num_samples)
sample_rewards, aux = reward_fn(sample_seqs, labels, config)
labels = sample_seqs[:, 0, ...]
rewards = sample_rewards[:, 0]
match_iou = aux["match_iou"][:, 0]
baselines = (jnp.sum(sample_rewards, axis=-1) - rewards) / (num_samples - 1)

# 2) Optimizize the model. By using REINFORCE to adjust the likelihood of the
# sequence based on the reward and with supervision to teach the model to
# predict the expected IoU of each box in its own samples.
def loss_fn(params):
  logits = model.apply(params, images, labels, train=True, rngs=rngs)
  logits_softmax = jax.nn.log_softmax(logits)

  # Use reinforce to optimize the expected reward for the whole sequence.
  seq_rewards = (rewards - baselines)
  # Note: consider improve this code to skip this loss for confidence tokens.
  # The paper did not do it due to a bug (and also does not seem to matter).
  target = jax.nn.one_hot(labels, logits.shape[-1]) * seq_rewards[:, None, None]
  loss_reward = -jnp.sum(target * logits_softmax, axis=-1)

  # Use supervision loss to tune the confidence tokens to predict IoU:
  #   - (1.0, 0.0, 0.0, ...)   -> for padded boxes.
  #   - (0.0, 1-iou, iou, ...) -> for sampled boxes.
  conf0 = (labels[:, 5::6] == 0)
  conf1 = (labels[:, 5::6] > 0) * (1.0 - match_iou)
  conf2 = (labels[:, 5::6] > 0) * match_iou
  target_conf = jnp.stack([conf0, conf1, conf2], axis=-1)
  logits_conf = logits_softmax[:, 5::6, :3]
  loss_conf = -jnp.sum(target_conf * logits_conf, axis=-1)

  loss = jnp.mean(loss_reward) + config.conf_w * jnp.mean(loss_conf)
  return loss
```
"""
import functools

import einops
import jax
import jax.numpy as jnp


# Frequency of COCO object detection classes as observed in the training set.
# pylint: disable=bad-whitespace,bad-continuation
CLS_COUNTS = [
    262465,  7113,  43867,   8725,   5135,   6069,   4571,   9973,  10759,
     12884,  1865,   1983,   1285,   9838,  10806,   4768,   5508,   6587,
      9509,  8147,   5513,   1294,   5303,   5131,   8720,  11431,  12354,
      6496,  6192,   2682,   6646,   2685,   6347,   9076,   3276,   3747,
      5543,  6126,   4812,  24342,   7913,  20650,   5479,   7770,   6165,
     14358,  9458,   5851,   4373,   6399,   7308,   7852,   2918,   5821,
      7179,  6353,  38491,   5779,   8652,   4192,  15714,   4157,   5805,
      4970,  2262,   5703,   2855,   6434,   1673,   3334,    225,   5610,
      2637, 24715,   6334,   6613,   1481,   4793,    198,   1954
]
# pylint: enable=bad-whitespace,bad-continuation


def seq2box(seq, max_level, max_conf, num_cls):
  """Extract boxes encoded as sequences."""
  # Reshape to instances of boxes
  dim_per_box = 6
  seq_len = seq.shape[-1]
  seq = seq[..., :(seq_len - seq_len % dim_per_box)]
  seq = einops.rearrange(seq, "... (n d) -> ... n d", d=dim_per_box)

  # Unpack box fields
  boxes, labels, confs = seq[..., 0:4], seq[..., 4], seq[..., 5]
  boxes = boxes - max_conf - 1
  labels = labels - max_conf - 1 - max_level - 1
  boxes = jnp.clip(boxes, 0, max_level) / max_level
  labels = jnp.clip(labels, 0, num_cls - 1)
  confs = jnp.clip(confs, 0, max_conf)

  return boxes, labels, confs


def iou_fn(box1, box2):
  """Compute IoU of two boxes."""
  ymin1, xmin1, ymax1, xmax1 = box1
  ymin2, xmin2, ymax2, xmax2 = box2

  a1 = jnp.abs((ymax1 - ymin1) * (xmax1 - xmin1))
  a2 = jnp.abs((ymax2 - ymin2) * (xmax2 - xmin2))

  yl = jnp.maximum(ymin1, ymin2)
  yr = jnp.minimum(ymax1, ymax2)
  yi = jnp.maximum(0, yr - yl)

  xl = jnp.maximum(xmin1, xmin2)
  xr = jnp.minimum(xmax1, xmax2)
  xi = jnp.maximum(0, xr - xl)

  inter = xi * yi
  return inter / (a1 + a2 - inter + 1e-9)

iou_fn_batched = jax.vmap(
    jax.vmap(iou_fn, in_axes=(None, 0)), in_axes=(0, None)
)


def _reward_fn_thr(seq_pred, seq_gt,
                   thr, nms_w, max_level, max_conf, num_cls, cls_smooth):
  """Compute detection reward function for a given IoU threshold."""
  # Weight matches of each label inversely proportional to the percentage of
  # GT instances with such label in the whole train dataset. Additionally
  # smooth out the observed distribution.
  cls_counts = jnp.array(CLS_COUNTS)
  weights = 1.0 / (cls_counts + cls_smooth*jnp.sum(cls_counts))
  weights = num_cls * weights / jnp.sum(weights)

  boxes_pred, labels_pred, confs_pred = seq2box(
      seq_pred, max_level, max_conf, num_cls)
  boxes_gt, labels_gt, confs_gt = seq2box(
      seq_gt, max_level, max_conf, num_cls)

  # Compute IoU matrix: Predictions X GT
  iou = iou_fn_batched(boxes_pred, boxes_gt)

  # IoU thr
  iou = jnp.where(iou > thr, iou, 0.0)

  # EOS mask
  confs_mask = (confs_pred[:, None] > 0) * (confs_gt[None, :] > 0)
  iou = confs_mask * iou

  # Label mask
  label_mask = labels_pred[:, None] == labels_gt[None, :]
  iou = label_mask * iou

  # Each prediction is matched to a single box
  single_match_mask = jax.nn.one_hot(jnp.argmax(iou, axis=1), iou.shape[1])
  iou = iou * single_match_mask

  # Pred. boxes indicators
  correct = jnp.any(iou > 0.0, axis=1).astype("int32") + 1
  correct = jnp.where(confs_pred > 0, correct, 0)

  # For each GT box find best match
  matches_idx = jnp.argmax(iou, axis=0)
  matches_iou = jnp.take_along_axis(iou, matches_idx[None], axis=0)[0]
  matches_idx = jnp.where(matches_iou > 0.0, matches_idx, -1)

  match_reward = jnp.sum((matches_idx >= 0) * weights[labels_gt][None, :])

  # Compute duplicate penalty (aka NMS).
  matches_mask = jax.nn.one_hot(matches_idx, iou.shape[0], axis=0)
  nms_penalty = jnp.sum(
      (iou > 0.0) * (1 - matches_mask) * weights[labels_pred][:, None])

  match_iou = jnp.sum(iou, axis=1)

  return {
      "reward": (match_reward - nms_w * nms_penalty),
      "num_matches": jnp.sum(matches_idx >= 0),
      "nms_penalty": nms_penalty,
      "correct": correct,
      "match_iou": match_iou,
  }


def reward_fn(seqs_pred, seqs_gt, config):
  """Total reward."""
  result = {}
  thrs = config.reward_thr
  correct_thr = config.correct_thr
  r_keys = ["reward", "num_matches", "nms_penalty"]
  for thr in thrs:
    fn = functools.partial(
        _reward_fn_thr,
        thr=thr,
        nms_w=config.nms_w,
        max_level=config.max_level,
        max_conf=config.max_conf,
        num_cls=config.num_cls,
        cls_smooth=config.cls_smooth,
    )
    rewards = jax.vmap(jax.vmap(fn, in_axes=(0, None)))(seqs_pred, seqs_gt)

    result = {**result, **{f"{k}-{thr:0.1f}": rewards[k]
                           for k in r_keys}}
    if thr == correct_thr:
      correct = rewards["correct"]
      match_iou = rewards["match_iou"]

  result = {
      **result,
      **{k: jnp.mean(
          jnp.array([result[f"{k}-{thr:0.1f}"] for thr in thrs]), axis=0)
         for k in r_keys}
  }

  return result["reward"], {
      "result": result,
      "correct": correct,
      "match_iou": match_iou,
  }
