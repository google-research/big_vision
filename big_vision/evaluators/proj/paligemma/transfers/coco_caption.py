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

"""Evaluator for caption generation metrics used for the MS COCO dataset."""
import collections
import functools
import os
import tempfile

import big_vision.evaluators.common as c
import big_vision.input_pipeline
import big_vision.pp.builder
import big_vision.pp.tokenizer
import big_vision.utils as u

from pycocoevalcap.bleu import bleu
from pycocoevalcap.cider import cider
from pycocoevalcap.meteor import meteor
from pycocoevalcap.rouge import rouge
from pycocoevalcap.spice import spice
from pycocoevalcap.tokenizer import ptbtokenizer

import jax

from tensorflow.io import gfile

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


class Evaluator:
  """Evaluator for caption generation metrics used for the MS COCO dataset.

  See https://arxiv.org/pdf/1504.00325.pdf or the repository implementing it
  https://github.com/tylin/coco-caption for details on the metrics. This code
  uses the python3 pip package from: https://github.com/salaniz/pycocoevalcap

  Note that both the model caption and the ground truth reference captions are
  further processed with the PTBTokenizer before computing scores.

  `predict_fn` accepts arbitrary dictionaries of parameters and data, where
  the data dictionary is produced by the `pp_fn` op. It is expected to output a
  dict containing tokenized captions.

  `pp_fn` must have fields: "image/id" and "captions".
  """

  def __init__(
      self, predict_fn, tokenizer=None,
      metrics=("cider",),  # Default to only cider. We often just look at that.
      preds_outfile="{workdir}/{name}_{split}_preds.json",
      annot_outfile="{workdir}/{name}_{split}_annotations.json",
      *, data, devices, **kw
  ):
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={"image/id", "captions"}, data=data, devices=devices, **kw)

    self.preds_outfile = c.resolve_outfile(
        preds_outfile, name=data.get("name"), split=data.get("split"))
    self.annot_outfile = c.resolve_outfile(
        annot_outfile, name=data.get("name"), split=data.get("split"))

    self.metrics = metrics
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token)

  def run(self, train_state):
    """Run eval."""
    gts = []
    res = []

    for _, batch in zip(range(self.steps), self.get_data_iter()):
      # (batch, seqlen) array of decoded generated tokens.
      tokens = self.decode(train_state, batch)

      # (local_batch,)
      tokens = u.get_local_slice_from_fsarray(tokens)
      ex_masks = u.get_local_slice_from_fsarray(batch["_mask"])

      image_ids = batch["image/id"][ex_masks]
      pred_captions = self.tok.to_str(tokens[ex_masks])

      for image_id, caption in zip(image_ids, pred_captions):
        res.append({"image_id": image_id.item(), "caption": caption})

      for image_id, captions in zip(image_ids, batch["captions"]):
        for caption in captions:
          gts.append({"image_id": image_id.item(), "caption": caption.item()})

    # Write model outputs following: https://cocodataset.org/#format-results
    # Use same format for gt although that is not the usual format for them.
    res = c.multiprocess_write_json(self.preds_outfile, res)
    gts = c.multiprocess_write_json(self.annot_outfile, gts)

    if jax.process_index():  # Host0 gets all preds and does eval.
      return

    outs = self.evaluate(gts, res)
    for key, score in outs.items():
      yield key, score

  def evaluate(self, gt_annotations, res_annotations):
    """Creates scorers and run evaluation."""
    scorers = {
        "rouge": rouge.Rouge,
        "cider": cider.Cider,
        "bleu-4": bleu.Bleu,
        "spice": spice.Spice,
        "meteor": meteor.Meteor,
    }

    # Reformat gts and res from [{"image_id": int|str, "caption": str}] to
    # {int_image_id: [{"caption": str}]} as expected by tokenizer and scorers.
    # Note there are multiple reference captions for the ground truth but only
    # one for the model predictions.
    iid_map = collections.defaultdict(lambda: len(iid_map))
    res = {iid_map[x["image_id"]]: [x] for x in res_annotations}
    gts = collections.defaultdict(list)
    for x in gt_annotations:
      gts[iid_map[x["image_id"]]].append(x)
    assert sorted(gts.keys()) == sorted(res.keys())

    # Tokenize captions and predictions using coco tokenizer.
    coco_tokenizer = ptbtokenizer.PTBTokenizer()
    gts = coco_tokenizer.tokenize(gts)
    res = coco_tokenizer.tokenize(res)

    scores = {}
    for metric in self.metrics:
      scorer = scorers[metric]()
      scores[metric], _ = scorer.compute_score(gts, res)
    return scores
