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

"""Evaluator for ChartQA variants."""

import functools

import big_vision.evaluators.common as c
import big_vision.pp.tokenizer
import big_vision.utils as u


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


class Evaluator:
  """Evaluator for simple VQA tasks."""

  def __init__(
      self, predict_fn, tokenizer, to_lower=False,
      outfile="{workdir}/{split}.json",
      out_question_key="question_id", out_answer_key="answer",
      *, data, devices, **kw):
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={"answer", "question_id"}, data=data, devices=devices, **kw)

    self.outfile = c.resolve_outfile(outfile, split=data.get("split"))
    self.out_question_key = out_question_key
    self.out_answer_key = out_answer_key

    # We'll need the tokenizer to detokenize the model outputs later.
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.postproc = (lambda s: s.lower()) if to_lower else lambda s: s
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token)

  def run(self, train_state):
    """Does one evaluation run, yields metrics."""

    accuracies = []
    relaxed_accuracies = []
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

        answer = self.postproc(self.tok.to_str(tokens[i], stop_at_eos=True))

        gt = self.postproc(batch["answer"][i])
        accuracies.append(float(answer == gt))
        relaxed_accuracies.append(_relaxed_match(gt, answer))
        json_out.append({
            self.out_question_key: batch["question_id"][i].item(),
            self.out_answer_key: answer,
            "gt": gt,
            "relaxed_match": relaxed_accuracies[-1],
        })

    # At this point `accuracies` is a list of per-example scores. However,
    # remember that each host holds a different subset of the examples! So if
    # we were to just return the mean accuracy here, we would effectively only
    # have evaluated on the main host's (who writes metrics) subset!
    # So now, we need to compute global means.
    # There is one more caveat: `process_sum` needs the summands on each host
    # to have the same size. So we either need to include dummy values for
    # the padding examples (last batch, annoying), or we only sum scalars as in
    # sufficient statistics, which we do here.
    sum_accs, sum_relaxed_accs, num = c.process_sum(
        [sum(accuracies), sum(relaxed_accuracies), len(accuracies)])

    # Yielding metric_name, value means logging the metric.
    yield "acc", sum_accs / num
    yield "relaxed_acc", sum_relaxed_accs / num
    yield "num", num  # Just for sanity checks.
    c.multiprocess_write_json(self.outfile, json_out)


def _to_float(text: str) -> float | None:
  try:
    if text.endswith("%"):
      # Convert percentages to floats.
      return float(text.rstrip("%")) / 100.0
    else:
      return float(text)
  except ValueError:
    return None


def _relaxed_match(target: str,
                   prediction: str,
                   max_relative_error: float = 0.05) -> bool:
  """Calculates relaxed correctness.

  The correctness tolerates certain error ratio defined by max_relative_error.
  See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
  “Following Methani et al. (2020), we use a relaxed accuracy measure for the
  numeric answers to allow a minor inaccuracy that may result from the automatic
  data extraction process. We consider an answer to be correct if it is within
  5% of the gold answer. For non-numeric answers, we still need an exact match
  to consider an answer to be correct.”

  Args:
    target: Target string.
    prediction: Predicted string.
    max_relative_error: Maximum relative error.

  Returns:
    Whether the prediction was correct given the specified tolerance.
  """
  prediction_float = _to_float(prediction)
  target_float = _to_float(target)
  # When the target is 0 is always required an exact match.
  if prediction_float is not None and target_float:
    relative_error = abs(prediction_float - target_float) / abs(target_float)
    return relative_error <= max_relative_error
  else:
    return prediction == target
