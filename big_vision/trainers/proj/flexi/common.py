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

"""Few common utils used in both/all flexi-trainers."""
import functools
import itertools
import numpy as np


def mkrng(xid, wid, step):
  # Need to cap at 0, for example localruns use -1.
  rng_key = (max(xid, 0), max(wid, 0), max(step, 0))
  return np.random.default_rng(rng_key)


def mkprob(x):
  if x is None:
    return x
  return np.array(x) / np.sum(x)


def choice(values, ratios, rng=None):
  rng = rng or np.random.default_rng()
  return rng.choice(values, p=mkprob(ratios))


def mkpredictfns(predict_fn, config, template="predict_{x}"):
  # If we have two flexi args a=[1,2], b=[10,20], then we create a
  # predict_fn for all possible combinations, named "predict_a=1_b=10" etc.
  all_combinations = [dict(comb) for comb in itertools.product(
      *[[(arg, val) for val in config[arg].v] for arg in config]
  )]
  return {
      template.format(x="_".join(f"{k}={v}" for k, v in kw.items())):
          functools.partial(predict_fn, **kw)
      for kw in all_combinations}
