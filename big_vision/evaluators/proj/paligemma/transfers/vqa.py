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

"""Evaluator for simple VQA variants (OCR-VQA, OKVQA, A-OKVQA).

According to the (A-)OKVAQ papers, the eval for these datasets should follow
VQAv2. But here we don't track different answer-types, and don't do any
leave-one-out averaging, as this isn't done in the official implementation at
https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py
either.

Please read the description of how evaluators work at (internal link).
This evaluator follows the pattern of also parallelizing the CPU computations
(ie postprocessing, score computation) across hosts for more scalability.

For now, simple decoding is implemented as part of the evaluator. We'll soon
unify and move to a library of decoding functions, including fancier and more
efficient ones.
"""
import functools

import big_vision.evaluators.common as c
import big_vision.pp.tokenizer
import big_vision.utils as u
import editdistance


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


class Evaluator:
  """Evaluator for simple VQA tasks.

  This evaluator expects the batch to contain a field `question_id` and a field
  `answer` for single ground truth or `answers` for multiple ground truths.

  The field names used when writting the json result can be controlled with
  `out_question_key` and `out_answer_key`.
  """

  def __init__(
      self, predict_fn, tokenizer, to_lower=False,
      outfile="{workdir}/{split}.json",
      out_question_key="question_id", out_answer_key="answer",
      *, data, devices, **kw):
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={"answers", "answer", "question_id"},
        data=data, devices=devices, **kw)

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
    accuracies_any = []
    anls_values = []
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

        # Now we have two commonly used VQA evaluation modes:
        if "answer" in batch:
          # single GT (eg ocrvqa): just compare to that answer, done.
          gt = self.postproc(batch["answer"][i])
          gts = [gt]
          accuracies.append(float(answer == gt))
          accuracies_any.append(float(answer == gt))
          anls_values.append(anls_metric(gt, answer))
        elif "answers" in batch and (gt_answers := batch["answers"][i]).size:
          # multiple GTs (eg okvqa): introduced by VQA, compare to each of them
          # with a threshold, see also: https://visualqa.org/evaluation.html
          gts = [self.postproc(a) for a in gt_answers]
          num_match = sum([answer == gt for gt in gts])
          accuracies.append(min(1.0, num_match / 3.0))
          accuracies_any.append(min(1.0, float(num_match)))
          anls_values.append(max(anls_metric(gt, answer) for gt in gts))
        else:
          gts = []

        json_out.append({
            self.out_question_key: batch["question_id"][i].item(),
            self.out_answer_key: answer} | ({"gts": gts} if gts else {}))

    # At this point `accuracies` is a list of per-example scores. However,
    # remember that each host holds a different subset of the examples! So if
    # we were to just return the mean accuracy here, we would effectively only
    # have evaluated on the main host's (who writes metrics) subset!
    # So now, we need to compute global means.
    # There is one more caveat: `process_sum` needs the summands on each host
    # to have the same size. So we either need to include dummy values for
    # the padding examples (last batch, annoying), or we only sum scalars as in
    # sufficient statistics, which we do here.
    sum_accs, sum_accs_any, sum_anls, num_accs, num = c.process_sum(
        [sum(accuracies), sum(accuracies_any), sum(anls_values),
         len(accuracies), len(json_out)])

    # Yielding metric_name, value means logging the metric.
    if num_accs:
      yield "acc", sum_accs / num_accs
      yield "acc_any", sum_accs_any / num_accs
      yield "anls", sum_anls / num_accs

    yield "num", num  # Just for sanity checks.
    c.multiprocess_write_json(self.outfile, json_out)


def anls_metric(target: str, prediction: str, theta: float = 0.5):
  """Calculates ANLS for DocVQA.

  There does not seem to be an official evaluation script.
  Public implementation on which this implementation is based:
  https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

  Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

  Args:
    target: Target string.
    prediction: Predicted string.
    theta: Filter threshold set to 0.5 for DocVQA.

  Returns:
    ANLS score.
  """
  if target:
    edit_distance = editdistance.eval(target, prediction)
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0
  else:
    return float(prediction == "")
