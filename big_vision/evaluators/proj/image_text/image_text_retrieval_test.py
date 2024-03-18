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

"""Unit tests for image_text_retrieval."""
from typing import Mapping

from absl.testing import absltest
from absl.testing import parameterized
from big_vision.evaluators.proj.image_text import image_text_retrieval
import numpy as np


class ImTextRetrievalTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.array([[0.0, 0.0, 0.1, 0.5, 0.1, 0.2, 0.5, 0.1],
                 [0.5, 0.4, 0.0, 0.0, 0.4, 0.2, 0.6, 0.4],
                 [0.5, 0.4, 0.1, 0.5, 0.0, 0.0, 0.8, 0.3],
                 [0.5, 0.4, 0.1, 0.5, 0.3, 0.2, 0.0, 0.0]]), {
                     'Recall@1': 1.0,
                     'Recall@5': 1.0,
                     'Recall@10': 1.0
                 }),  #
      (np.array([[0.8, 0.8, 0.1, 0.5, 0.1, 0.2, 0.5, 0.1],
                 [0.5, 0.4, 0.0, 0.0, 0.4, 0.2, 0.6, 0.4],
                 [0.5, 0.4, 0.1, 0.5, 0.0, 0.8, 0.8, 0.3],
                 [0.5, 0.4, 0.1, 0.5, 0.4, 0.2, 0.3, 0.3]]), {
                     'Recall@1': 0.5,
                     'Recall@5': 0.75,
                     'Recall@10': 1.0
                 }))
  def test_image_to_text_retrieval_eval(self, dist_matrix: np.ndarray,
                                        expected: Mapping[str, float]):
    """Checks `image_to_text_retrieval_eval`.

    Args:
      dist_matrix: Distance matrix between image (rows) and text (columns).
      expected: Expected eval results.
    """
    self.assertEqual(
        image_text_retrieval.image_to_text_retrieval_eval(
            dist_matrix, [0, 0, 1, 1, 2, 2, 3, 3]), expected)

  @parameterized.parameters(
      (np.array([[0.0, 0.0, 0.1, 0.5, 0.1, 0.2, 0.5, 0.1],
                 [0.5, 0.4, 0.0, 0.0, 0.4, 0.2, 0.6, 0.4],
                 [0.5, 0.4, 0.1, 0.5, 0.0, 0.0, 0.8, 0.3],
                 [0.5, 0.4, 0.1, 0.5, 0.3, 0.2, 0.0, 0.0]]), {
                     'Recall@1': 1.0,
                     'Recall@5': 1.0,
                     'Recall@10': 1.0
                 }),  #
      (np.array([[0.8, 0.8, 0.1, 0.5, 0.1, 0.2, 0.1, 0.1],
                 [0.5, 0.4, 0.0, 0.0, 0.4, 0.2, 0.6, 0.4],
                 [0.5, 0.4, 0.1, 0.5, 0.0, 0.8, 0.8, 0.3],
                 [0.5, 0.4, 0.1, 0.5, 0.4, 0.2, 0.3, 0.3]]), {
                     'Recall@1': 0.375,
                     'Recall@5': 1.0,
                     'Recall@10': 1.0
                 }))
  def test_image_text_retrieval(self, dist_matrix: np.ndarray,
                                expected: Mapping[str, float]):
    """Checks `text_to_image_retrieval_eval`.

    Args:
      dist_matrix: Distance matrix between image (rows) and text (columns).
      expected: Expected eval results.
    """
    self.assertEqual(
        image_text_retrieval.text_to_image_retrieval_eval(
            dist_matrix, [0, 0, 1, 1, 2, 2, 3, 3]), expected)


if __name__ == '__main__':
  absltest.main()
