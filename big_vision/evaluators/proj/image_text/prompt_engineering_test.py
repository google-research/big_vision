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

"""Tests for prompt_engineering."""

from absl.testing import absltest
from big_vision.evaluators.proj.image_text import prompt_engineering


class PromptEngineeringTest(absltest.TestCase):

  def test_canonicalize_text(self):
    self.assertEqual(prompt_engineering.canonicalize_text("test_test"), "test test")
    self.assertEqual(
        prompt_engineering.canonicalize_text("test___test"), "test test")
    self.assertEqual(prompt_engineering.canonicalize_text("test"), "test")
    self.assertEqual(prompt_engineering.canonicalize_text("test."), "test")
    self.assertEqual(prompt_engineering.canonicalize_text(" test "), "test")
    self.assertEqual(
        prompt_engineering.canonicalize_text("test\ntest"), "test test")
    self.assertEqual(
        prompt_engineering.canonicalize_text("test  test"), "test test")
    self.assertEqual(prompt_engineering.canonicalize_text("test {}"), "test")
    self.assertEqual(
        prompt_engineering.canonicalize_text(
            "test {}", keep_punctuation_exact_string="{}"), "test {}")
    self.assertEqual(
        prompt_engineering.canonicalize_text(
            " test  {}...", keep_punctuation_exact_string="{}"), "test {}")
    self.assertEqual(
        prompt_engineering.canonicalize_text(
            "test {}  {}  {}", keep_punctuation_exact_string="{}"),
        "test {} {} {}")


if __name__ == "__main__":
  absltest.main()
