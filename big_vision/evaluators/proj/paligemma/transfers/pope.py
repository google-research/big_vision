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

"""Evaluator for the POPE dataset (https://github.com/RUCAIBox/POPE).

POPE is a binary classification dataset with ground-truth answers being either
'yes' or 'no'.
"""

import functools

import big_vision.datasets.core
import big_vision.evaluators.common as c
import big_vision.input_pipeline
import big_vision.pp.builder
import big_vision.pp.tokenizer
import big_vision.utils as u


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


class Evaluator:
  """Evaluator for the POPE task.

  This evaluator expects the batch to contain a field `question_id` and a field
  `answer` for single ground truth or `answers` for multiple ground truths.

  The field names used when writting the json result can be controlled with
  `out_question_key` and `out_answer_key`.
  """

  def __init__(
      self,
      predict_fn,
      data,
      pp_fn,
      tokenizer,
      batch_size,
      *,
      devices,
      outfile="{workdir}/{split}.json",
      out_question_key="question_id",
      out_answer_key="answer"
  ):

    self.outfile = c.resolve_outfile(outfile, split=data.get("split"))
    self.out_question_key = out_question_key
    self.out_answer_key = out_answer_key
    # This will mostly look the same across all evaluators, preparing data:
    data = big_vision.datasets.core.get(**data)
    pp_fn = big_vision.pp.builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = big_vision.input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True),
        pp_fn,
        batch_size,
        num_ex_per_process=data.num_examples_per_process(),
    )
    # The `keep_on_cpu=` argument lists the data keys that, if they exist, we
    # do NOT want to ship to the TPUs and instead just keep in host memory.
    # Typically ground-truth and metadata, that is often of string type.
    self.data_iter = big_vision.input_pipeline.start_global(
        self.ds, devices, keep_on_cpu={"answer", "question_id"}
    )
    # We'll need the tokenizer to detokenize the model outputs later.
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token
    )

  def run(self, train_state):
    """Does one evaluation run, yields metrics."""

    accuracies = []
    valid = []
    json_out = []
    for _, batch in zip(range(self.steps), self.data_iter):
      # (batch, seqlen) array of decoded generated tokens.
      tokens = self.decode(train_state, batch)

      # (local_batch,) that indicates padding examples (0) vs real examples (1).
      tokens = u.get_local_slice_from_fsarray(tokens)
      ex_masks = u.get_local_slice_from_fsarray(batch["_mask"])

      # Turn predictions into texts and then scores, one by one.
      for i in range(len(tokens)):
        if ex_masks[i] == 0:  # Skip last-batch padding examples
          continue

        answer = self.tok.to_str(tokens[i], stop_at_eos=True).lower()
        gt = batch["answer"][i]
        accuracies.append(float(answer == gt))
        valid.append(float(answer in ("yes", "no")))

        json_out.append(
            {
                self.out_question_key: batch["question_id"][i].item(),
                self.out_answer_key: answer,
            }
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
    sum_accs, sum_valid, num = c.process_sum([
        sum(accuracies),
        sum(valid),
        len(accuracies),
    ])

    if num:
      yield "acc", sum_accs / num
      yield "valid_percent", sum_valid / num
    yield "num", num

    c.multiprocess_write_json(self.outfile, json_out)
