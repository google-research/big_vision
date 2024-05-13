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

"""Evaluator for VQAV2 dataset.
"""
import functools
import re

import big_vision.evaluators.common as c
import big_vision.pp.tokenizer
import big_vision.utils as u
import numpy as np


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


class Evaluator:
  """VQAv2 evaluator."""

  def __init__(
      self, predict_fn, tokenizer, outfile="{workdir}/{split}.json",
      *, data, devices, **kw):
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu={"answers", "answer_type", "question_type", "question_id"},
        data=data, devices=devices, **kw)

    self.outfile = c.resolve_outfile(outfile, split=data.get("split"))

    # We'll need the tokenizer to detokenize the model outputs later.
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token)

  def run(self, train_state):
    """Does one evaluation run, yields metrics."""
    accuracies_by_type = {"yes/no": [], "number": [], "other": []}
    json_out = []

    for _, batch in zip(range(self.steps), self.get_data_iter()):
      # (batch, seqlen) array of decoded (generated) token sequences suffixes.
      tokens = self.decode(train_state, batch)

      # (local_batch,) that indicates padding examples (0) vs real examples (1).
      tokens = u.get_local_slice_from_fsarray(tokens)
      ex_masks = u.get_local_slice_from_fsarray(batch["_mask"])

      # Turn predictions into texts and then scores, one by one.
      for i in range(len(tokens)):
        if ex_masks[i] == 0:  # Skip last-batch padding examples
          continue

        # Extract the suffix/answer from the generated string, skip bos.
        answer = self.tok.to_str(tokens[i], stop_at_eos=True)
        json = {"question_id": batch["question_id"][i].item(), "answer": answer}

        # The rest is computation of VQA-score which compares to multiple GTs.
        # This is described better here: https://visualqa.org/evaluation.html
        if (gt_answers := batch["answers"][i]).size:
          # Always need to do light space-processing:
          gt_answers = [stripspace_vqav2(a) for a in gt_answers]
          answer = stripspace_vqav2(answer)

          # Only post-process if not all agree. Supposedly avoids postproc OCR:
          # https://github.com/GT-Vision-Lab/VQA/issues/14#issuecomment-1334695361
          if len(set(gt_answers)) > 1:
            answer = postprocess_vqav2_text(answer)
            gt_answers = [postprocess_vqav2_text(a) for a in gt_answers]

          # Accuracy is avg over all ten leave-one-out GT's.
          # https://github.com/GT-Vision-Lab/VQA/issues/1#issuecomment-199921352
          # An answer is counted 100% correct as soon as 3 GT's agree with it.
          matches = answer == np.array(gt_answers)
          acc = np.mean([
              np.clip(np.sum(np.delete(matches, i_leave_out)) / 3, 0, 1)
              for i_leave_out in range(10)
          ])

          accuracies_by_type[batch["answer_type"][i]].append(acc)

          # Update json with fully post-processed answer and gt:
          json["answer_raw"] = json["answer"]
          json["answer"] = answer
          json["gts"] = gt_answers

        json_out.append(json)

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
    num = c.process_sum(len(json_out))

    # Yielding metric_name, value means logging the metric.
    if n := sum(num_accs.values()):
      yield "acc", sum(sum_accs.values()) / n
    if n := num_accs["yes/no"]:
      yield "acc/yesno", sum_accs["yes/no"] / n
      yield "num/yesno", n
    if n := num_accs["number"]:
      yield "acc/number", sum_accs["number"] / n
      yield "num/number", n
    if n := num_accs["other"]:
      yield "acc/other", sum_accs["other"] / n
      yield "num/other", n

    yield "num", num  # Just for sanity checks.
    c.multiprocess_write_json(self.outfile, json_out)


# Post-processing required is described at https://visualqa.org/evaluation.html


def stripspace_vqav2(txt):
  return txt.replace("\n", " ").replace("\t", " ").strip()


def postprocess_vqav2_text(txt):
  """Cleanup string according to VQA."""
  has_digit_comma = re.search(r"(\d)(\,)(\d)", txt) is not None

  out = txt
  for p in PUNCT:
    # NOTE: digit_comma here looks like a bug in official code, so we follow it.
    if has_digit_comma or f"{p} " in txt or f" {p}" in txt:
      out = out.replace(p, "")
    else:
      out = out.replace(p, " ")

  # Remove full-stops that aren't part of a number.
  out = re.sub(r"(?!<=\d)(\.)(?!\d)", "", out, flags=re.UNICODE)

  words = []
  for word in out.lower().split():
    if word not in ARTICLES:
      words.append(REPLACEMENTS.get(word, word))
  return " ".join(words)


# pylint: disable=line-too-long
REPLACEMENTS = {
    # CONTRACTIONS
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
    # NUMBERS
    "none": "0", "zero": "0", "one": "1", "two": "2",
    "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
# pylint: enable=line-too-long

PUNCT = [
    ";", "/", "[", "]", "\"", "{", "}",
    "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!"
]
ARTICLES = {"a", "an", "the"}
