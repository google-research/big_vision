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

"""Evaluator for ScienceQA.

based on the official implementation at
https://github.com/lupantech/ScienceQA/blob/main/models/run_gpt3.py
"""

import functools
import re

import big_vision.evaluators.common as c
import big_vision.pp.tokenizer
import big_vision.utils as u


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"
FAILURE = "failed"


class Evaluator:
  """Evaluator for simple VQA tasks."""

  def __init__(
      self, predict_fn, tokenizer,
      outfile="{workdir}/{split}.json",
      out_question_key="question_id",
      *, data, devices, **kw):
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={"answer", "question_id"}, data=data, devices=devices, **kw)

    self.outfile = c.resolve_outfile(outfile, split=data.get("split"))
    self.out_question_key = out_question_key

    # We'll need the tokenizer to detokenize the model outputs later.
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token
    )

  def postproc(self, raw_answer):
    """Post-processes the raw answer. extract a, b, c from the string."""
    match = re.match(
        pattern=r"the answer is ([a-z])\.", string=raw_answer.lower()
    )
    if match:
      return match.groups()[0]  # 'a', 'b', ...
    else:
      return FAILURE

  def run(self, train_state):
    """Does one evaluation run, yields metrics."""

    accuracies = []
    fail_parse = []
    json_out = []
    for _, batch in zip(range(self.steps), self.get_data_iter()):
      # (batch, seqlen) array of decoded generated tokens.
      tokens = self.decode(train_state, batch)

      # (local_batch,) that indicates padding examples (0) vs real examples (1).
      tokens = u.get_local_slice_from_fsarray(tokens)
      ex_masks = u.get_local_slice_from_fsarray(batch["_mask"])

      # Turn predictions into texts and then scores, one by one.
      for i in range(len(tokens)):
        if ex_masks[i] == 0:  # Skip last-batch padding examples
          continue

        raw_answer = self.tok.to_str(tokens[i], stop_at_eos=True)
        answer = self.postproc(raw_answer)
        if "answer" in batch:
          gt = self.postproc(batch["answer"][i])
          gts = [gt]
          accuracies.append(float(answer == gt))
          fail_parse.append(float(answer == FAILURE))
        else:
          gts = []

        json_out.append(
            {
                self.out_question_key: batch["question_id"][i].item(),
                "raw_answer": raw_answer,
                "answer": answer,
            }
            | ({"gts": gts} if gts else {})
        )

    # At this point `accuracies` is a list of per-example scores. However,
    # remember that each host holds a different subset of the examples! So if
    # we were to just return the mean accuracy here, we would effectively only
    # have evaluated on the main host's (who writes metrics) subset!
    # So now, we need to compute global means.
    # There is one more caveat: `process_sum` needs the summands on each host
    # to have the same size. So we either need to include dummy values for
    # the padding examples (last batch, annoying), or we only sum scalars as in
    # sufficient statistics, which we do here.
    sum_accs, num_parsefail, num_accs, num = c.process_sum(
        [sum(accuracies), sum(fail_parse), len(accuracies), len(json_out)]
    )

    # Yielding metric_name, value means logging the metric.
    if num_accs > 0:
      yield "acc", sum_accs / num_accs
      yield "parsefail", num_parsefail / num_accs

    yield "num", num  # Just for sanity checks.
    c.multiprocess_write_json(self.outfile, json_out)
