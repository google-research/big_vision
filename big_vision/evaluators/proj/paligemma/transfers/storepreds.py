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

"""Evaluator to run inference and store results."""
import functools

import big_vision.evaluators.common as c
import big_vision.input_pipeline
import big_vision.pp.builder
import big_vision.pp.tokenizer
import big_vision.utils as u

import jax

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


class Evaluator:
  """Evaluator to run inference and store results."""

  def __init__(
      self, predict_fn, tokenizer=None,
      preds_outfile="{workdir}/{name}_{split}_preds.json",
      annot_outfile="{workdir}/{name}_{split}_annotations.json",
      id_key="id",
      *, data, devices, **kw
  ):
    self.id_key = id_key
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={id_key}, data=data, devices=devices, **kw)

    self.preds_outfile = c.resolve_outfile(
        preds_outfile, name=data.get("name"), split=data.get("split", ""))
    self.annot_outfile = c.resolve_outfile(
        annot_outfile, name=data.get("name"), split=data.get("split", ""))

    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token)

  def run(self, train_state):
    """Run eval."""
    res = []

    for _, batch in zip(range(self.steps), self.get_data_iter()):
      # (batch, seqlen) array of decoded generated tokens.
      tokens = self.decode(train_state, batch)

      # (local_batch,)
      tokens = u.get_local_slice_from_fsarray(tokens)
      ex_masks = u.get_local_slice_from_fsarray(batch["_mask"])

      image_ids = batch[self.id_key][ex_masks]
      pred_captions = self.tok.to_str(tokens[ex_masks])

      for image_id, caption in zip(image_ids, pred_captions):
        res.append({self.id_key: str(image_id), "caption": caption})

    res = c.multiprocess_write_json(self.preds_outfile, res)

    if jax.process_index():  # Host0 gets all preds and does eval.
      return

    yield "num_examples", len(res)
