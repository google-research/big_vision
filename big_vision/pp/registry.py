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

"""Global Registry for big_vision pp ops.

Author: Joan Puigcerver (jpuigcerver@)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import contextlib
import functools


def parse_name(string_to_parse):
  """Parses input to the registry's lookup function.

  Args:
    string_to_parse: can be either an arbitrary name or function call
      (optionally with positional and keyword arguments).
      e.g. "multiclass", "resnet50_v2(filters_factor=8)".

  Returns:
    A tuple of input name, argument tuple and a keyword argument dictionary.
    Examples:
      "multiclass" -> ("multiclass", (), {})
      "resnet50_v2(9, filters_factor=4)" ->
          ("resnet50_v2", (9,), {"filters_factor": 4})

  Author: Joan Puigcerver (jpuigcerver@)
  """
  expr = ast.parse(string_to_parse, mode="eval").body  # pytype: disable=attribute-error
  if not isinstance(expr, (ast.Attribute, ast.Call, ast.Name)):
    raise ValueError(
        "The given string should be a name or a call, but a {} was parsed from "
        "the string {!r}".format(type(expr), string_to_parse))

  # Notes:
  # name="some_name" -> type(expr) = ast.Name
  # name="module.some_name" -> type(expr) = ast.Attribute
  # name="some_name()" -> type(expr) = ast.Call
  # name="module.some_name()" -> type(expr) = ast.Call

  if isinstance(expr, ast.Name):
    return string_to_parse, (), {}
  elif isinstance(expr, ast.Attribute):
    return string_to_parse, (), {}

  def _get_func_name(expr):
    if isinstance(expr, ast.Attribute):
      return _get_func_name(expr.value) + "." + expr.attr
    elif isinstance(expr, ast.Name):
      return expr.id
    else:
      raise ValueError(
          "Type {!r} is not supported in a function name, the string to parse "
          "was {!r}".format(type(expr), string_to_parse))

  def _get_func_args_and_kwargs(call):
    args = tuple([ast.literal_eval(arg) for arg in call.args])
    kwargs = {
        kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in call.keywords
    }
    return args, kwargs

  func_name = _get_func_name(expr.func)
  func_args, func_kwargs = _get_func_args_and_kwargs(expr)

  return func_name, func_args, func_kwargs


class Registry(object):
  """Implements global Registry.

  Authors: Joan Puigcerver (jpuigcerver@), Alexander Kolesnikov (akolesnikov@)
  """

  _GLOBAL_REGISTRY = {}

  @staticmethod
  def global_registry():
    return Registry._GLOBAL_REGISTRY

  @staticmethod
  def register(name, replace=False):
    """Creates a function that registers its input."""

    def _register(item):
      if name in Registry.global_registry() and not replace:
        raise KeyError("The name {!r} was already registered.".format(name))

      Registry.global_registry()[name] = item
      return item

    return _register

  @staticmethod
  def lookup(lookup_string, kwargs_extra=None):
    """Lookup a name in the registry."""

    try:
      name, args, kwargs = parse_name(lookup_string)
    except ValueError as e:
      raise ValueError(f"Error parsing:\n{lookup_string}") from e
    if kwargs_extra:
      kwargs.update(kwargs_extra)
    item = Registry.global_registry()[name]
    return functools.partial(item, *args, **kwargs)

  @staticmethod
  def knows(lookup_string):
    try:
      name, _, _ = parse_name(lookup_string)
    except ValueError as e:
      raise ValueError(f"Error parsing:\n{lookup_string}") from e
    return name in Registry.global_registry()


@contextlib.contextmanager
def temporary_ops(**kw):
  """Registers specified pp ops for use in a `with` block.

  Example use:

    with pp_registry.remporary_ops(
        pow=lambda alpha: lambda d: {k: v**alpha for k, v in d.items()}):
      pp = pp_builder.get_preprocess_fn("pow(alpha=2.0)|pow(alpha=0.5)")
      features = pp(features)

  Args:
    **kw: Names are preprocess string function names to be used to specify the
      preprocess function. Values are functions that can be called with params
      (e.g. the `alpha` param in above example) and return functions to be used
      to transform features.

  Yields:
    A context manager to be used in a `with` statement.
  """
  reg = Registry.global_registry()
  kw = {f"preprocess_ops.{k}": v for k, v in kw.items()}
  for k in kw:
    assert k not in reg
  for k, v in kw.items():
    reg[k] = v
  try:
    yield
  finally:
    for k in kw:
      del reg[k]
