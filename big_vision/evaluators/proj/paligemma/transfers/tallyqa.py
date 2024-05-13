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

"""Evaluator for TallyQA dataset."""

import functools

import big_vision.evaluators.common as c
import big_vision.pp.tokenizer
import big_vision.utils as u


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


# Largest count we want to track.
_LARGEST_COUNT = 15


class Evaluator:
  """TallyQA evaluator."""

  def __init__(self, predict_fn, tokenizer, *, devices, **kw):
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={"answer", "issimple"}, devices=devices, **kw)

    # We'll need the tokenizer to detokenize the model outputs later.
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token
    )

  def run(self, train_state):
    """Does one evaluation run, yields metrics."""

    accuracies_by_type = {"all": [], "simple": [], "complex": []}
    # Add per-count entries. Cannot use a `defaultdict` as we need to `tree_map`
    # over keys later in `c.process_sum`.
    accuracies_by_type.update(
        {f"count_{i}": [] for i in range(_LARGEST_COUNT + 1)}
    )

    for _, batch in zip(range(self.steps), self.get_data_iter()):
      # (batch, seqlen) array of decoded (generated) token sequences suffixes.
      tokens = self.decode(train_state, batch)

      # (local_batch,) that indicates padding examples (0) vs real examples (1).
      tokens = u.get_local_slice_from_fsarray(tokens)
      ex_masks = u.get_local_slice_from_fsarray(batch["_mask"])

      # Turn predictions into texts and then scores, one by one.
      # We always compare the gt (string digit, e.g. "1") to the answer by the
      # model (e.g. "1").
      for i in range(len(tokens)):
        if ex_masks[i] == 0:  # Skip last-batch padding examples
          continue

        # Extract the suffix/answer from the generated string, skip bos.
        answer = self.tok.to_str(tokens[i], stop_at_eos=True)
        # Standardize the reponse, i.e., convert number words ("one") to
        # numerals ("1").
        answer = _number_word_to_numeral(answer)

        # Always need to do light space-processing:
        gt = _number_word_to_numeral(batch["answer"][i])
        accuracies_by_type["all"].append(float(answer == gt))

        if "issimple" in batch:
          # Simple/complex split.
          if batch["issimple"][i] == 1:
            accuracies_by_type["simple"].append(float(answer == gt))
          elif batch["issimple"][i] == 0:
            accuracies_by_type["complex"].append(float(answer == gt))
          else:
            # Train set is not annotated with simple/complex (but has dummy
            # value of `-1` in this field).
            pass

        # Store accuracies per count.
        accuracies_by_type[f"count_{gt}"].append(float(answer == gt))

    # At this point `accuracies` is a list of per-example scores. However,
    # remember that each host holds a different subset of the examples! So if
    # we were to just return the mean accuracy here, we would effectively only
    # have evaluated on the main host's (who writes metrics) subset!
    # So now, we need to compute global means.
    # There is one more caveat: `process_sum` needs the summands on each host
    # to have the same size. So we either need to include dummy values for
    # the padding examples (last batch, annoying), or we only sum scalars as in
    # sufficient statistics, which we do here.
    sum_accs = c.process_sum({k: sum(v) for k, v in accuracies_by_type.items()})
    num_accs = c.process_sum({k: len(v) for k, v in accuracies_by_type.items()})

    if n := num_accs["all"]:
      yield "acc", sum_accs["all"] / n
      yield "num", n  # Just for sanity checks.
    for key in sum_accs.keys():
      if (key != "all") and (num_accs[key]):
        yield f"acc/{key}", sum_accs[key] / num_accs[key]
        yield f"num/{key}", num_accs[key]  # Just for sanity checks.


def _number_word_to_numeral(s: str) -> str:
  """Returns numeral for a given number word, e.g., "one" -> "1" (up to 20)."""
  return REPLACEMENTS.get(s.lower(), s)


REPLACEMENTS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
