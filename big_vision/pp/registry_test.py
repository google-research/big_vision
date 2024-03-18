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

"""Tests for registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import mock

from absl.testing import absltest
from big_vision.pp import registry


class RegistryTest(absltest.TestCase):

  def setUp(self):
    super(RegistryTest, self).setUp()
    # Mock global registry in each test to keep them isolated and allow for
    # concurrent tests.
    self.addCleanup(mock.patch.stopall)
    self.global_registry = dict()
    self.mocked_method = mock.patch.object(
        registry.Registry, "global_registry",
        return_value=self.global_registry).start()

  def test_parse_name(self):
    name, args, kwargs = registry.parse_name("f")
    self.assertEqual(name, "f")
    self.assertEqual(args, ())
    self.assertEqual(kwargs, {})

    name, args, kwargs = registry.parse_name("f()")
    self.assertEqual(name, "f")
    self.assertEqual(args, ())
    self.assertEqual(kwargs, {})

    name, args, kwargs = registry.parse_name("func(a=0,b=1,c='s')")
    self.assertEqual(name, "func")
    self.assertEqual(args, ())
    self.assertEqual(kwargs, {"a": 0, "b": 1, "c": "s"})

    name, args, kwargs = registry.parse_name("func(1,'foo',3)")
    self.assertEqual(name, "func")
    self.assertEqual(args, (1, "foo", 3))
    self.assertEqual(kwargs, {})

    name, args, kwargs = registry.parse_name("func(1,'2',a=3,foo='bar')")
    self.assertEqual(name, "func")
    self.assertEqual(args, (1, "2"))
    self.assertEqual(kwargs, {"a": 3, "foo": "bar"})

    name, args, kwargs = registry.parse_name("foo.bar.func(a=0,b=(1),c='s')")
    self.assertEqual(name, "foo.bar.func")
    self.assertEqual(kwargs, dict(a=0, b=1, c="s"))

    with self.assertRaises(SyntaxError):
      registry.parse_name("func(0")
    with self.assertRaises(SyntaxError):
      registry.parse_name("func(a=0,,b=0)")
    with self.assertRaises(SyntaxError):
      registry.parse_name("func(a=0,b==1,c='s')")
    with self.assertRaises(ValueError):
      registry.parse_name("func(a=0,b=undefined_name,c='s')")

  def test_register(self):
    # pylint: disable=unused-variable
    @registry.Registry.register("func1")
    def func1():
      pass

    self.assertLen(registry.Registry.global_registry(), 1)

  def test_lookup_function(self):

    @registry.Registry.register("func1")
    def func1(arg1, arg2, arg3):  # pylint: disable=unused-variable
      return arg1, arg2, arg3

    self.assertTrue(callable(registry.Registry.lookup("func1")))
    self.assertEqual(registry.Registry.lookup("func1")(1, 2, 3), (1, 2, 3))
    self.assertEqual(
        registry.Registry.lookup("func1(arg3=9)")(1, 2), (1, 2, 9))
    self.assertEqual(
        registry.Registry.lookup("func1(arg2=9,arg1=99)")(arg3=3), (99, 9, 3))
    self.assertEqual(
        registry.Registry.lookup("func1(arg2=9,arg1=99)")(arg1=1, arg3=3),
        (1, 9, 3))

    self.assertEqual(
        registry.Registry.lookup("func1(1)")(1, 2), (1, 1, 2))
    self.assertEqual(
        registry.Registry.lookup("func1(1)")(arg3=3, arg2=2), (1, 2, 3))
    self.assertEqual(
        registry.Registry.lookup("func1(1, 2)")(3), (1, 2, 3))
    self.assertEqual(
        registry.Registry.lookup("func1(1, 2)")(arg3=3), (1, 2, 3))
    self.assertEqual(
        registry.Registry.lookup("func1(1, arg2=2)")(arg3=3), (1, 2, 3))
    self.assertEqual(
        registry.Registry.lookup("func1(1, arg3=2)")(arg2=3), (1, 3, 2))
    self.assertEqual(
        registry.Registry.lookup("func1(1, arg3=2)")(3), (1, 3, 2))

    with self.assertRaises(TypeError):
      registry.Registry.lookup("func1(1, arg2=2)")(3)
    with self.assertRaises(TypeError):
      registry.Registry.lookup("func1(1, arg3=3)")(arg3=3)
    with self.assertRaises(TypeError):
      registry.Registry.lookup("func1(1, arg3=3)")(arg1=3)
    with self.assertRaises(SyntaxError):
      registry.Registry.lookup("func1(arg1=1, 3)")(arg2=3)


if __name__ == "__main__":
  absltest.main()
