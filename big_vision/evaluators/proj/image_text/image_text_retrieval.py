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

"""Evaluates image-text retrieval results."""
from typing import List, Mapping

import numpy as np

RECALL_THRESHOLDS = (1, 5, 10)


def text_to_image_retrieval_eval(
    dist_matrix: np.ndarray,
    text_image_correspondence: List[int]) -> Mapping[str, float]:
  """Runs the text-to-image retrieval eval from the distance matrix.

  Args:
    dist_matrix: Distance matrix between text and image embeddings (shape
      N_IMAGES x N_TEXTS).
    text_image_correspondence: Mapping between rows and columns of
      `dist_matrix`, that is, a list of N_TEXTS integers n_i that represent that
      the text embedding in column i corresponds to the image embedding in row
      n_i. Please note that many texts can be assigned to the same image. For
      instance, if we have 2 images and 4 texts (i.e. dist_matrix is 2x4), then
      `text_image_correspondence = [0, 0, 1, 1]` means that the two first texts
      correspond to the first image and the two last texts to the second image.

  Returns:
    A dictionary with the Recall@k scores for k in RECALL_THRESHOLDS.
  """
  per_text_ranks = dist_matrix.argsort(axis=0)
  text_image_correspondence = np.array(text_image_correspondence)

  def recall_at(k):
    wins = per_text_ranks[:k, :] == text_image_correspondence[None]
    return wins.any(axis=0).mean()

  return {
      f'Recall@{k}': recall_at(k)
      for k in RECALL_THRESHOLDS
  }


def image_to_text_retrieval_eval(
    dist_matrix: np.ndarray,
    text_image_correspondence: List[int]) -> Mapping[str, float]:
  """Runs the image-to-text retrieval eval from the distance matrix.

  Args:
    dist_matrix: Distance matrix between text and image embeddings (shape
      N_IMAGES x N_TEXTS).
    text_image_correspondence: Mapping between rows and columns of
      `dist_matrix`, that is, a list of N_TEXTS integers n_i that represent that
      the text embedding in column i corresponds to the image embedding in row
      n_i. Please note that many texts can be assigned to the same image. For
      instance, if we have 2 images and 4 texts (i.e. dist_matrix is 2x4), then
      `text_image_correspondence = [0, 0, 1, 1]` means that the two first texts
      correspond to the first image and the two last texts to the second image.

  Returns:
    A dictionary with the Recall@k scores for k in RECALL_THRESHOLDS.
  """
  per_image_ranks = dist_matrix.argsort(axis=1)
  text_image_correspondence = np.array(text_image_correspondence)

  def recall_at(k):
    top_k_images = text_image_correspondence[per_image_ranks[:, :k]]
    wins = top_k_images == np.arange(len(per_image_ranks))[:, None]
    return wins.any(axis=1).mean()

  return {
      f'Recall@{k}': recall_at(k)
      for k in RECALL_THRESHOLDS
  }
