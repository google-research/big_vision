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

"""Preprocessing utils."""

from collections import abc


def maybe_repeat(arg, n_reps):
  if not isinstance(arg, abc.Sequence) or isinstance(arg, str):
    arg = (arg,) * n_reps
  return arg


class InKeyOutKey(object):
  """Decorator for preprocessing ops, which adds `inkey` and `outkey` arguments.

  Note: Only supports single-input single-output ops.
  """

  def __init__(self, indefault="image", outdefault="image", with_data=False):
    self.indefault = indefault
    self.outdefault = outdefault
    self.with_data = with_data

  def __call__(self, orig_get_pp_fn):

    def get_ikok_pp_fn(*args, key=None,
                       inkey=self.indefault, outkey=self.outdefault, **kw):

      orig_pp_fn = orig_get_pp_fn(*args, **kw)
      def _ikok_pp_fn(data):
        # Optionally allow the function to get the full data dict as aux input.
        if self.with_data:
          data[key or outkey] = orig_pp_fn(data[key or inkey], data=data)
        else:
          data[key or outkey] = orig_pp_fn(data[key or inkey])
        return data

      return _ikok_pp_fn

    return get_ikok_pp_fn
