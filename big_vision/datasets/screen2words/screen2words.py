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

# pylint: disable=line-too-long
r"""Creates TFDS dataset for Screen2words.


Preparing the data:
  1) mkdir /tmp/data/rico && cd /tmp/data/rico
  2) wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
  3) tar xvfz unique_uis.tar.gz && rm unique_uis.tar.gz
  4) git clone https://github.com/google-research-datasets/screen2words.git

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=screen2words

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('screen2_words', split='train', data_dir='/tmp/tfds')
"""
# pylint: enable=line-too-long
import collections
import csv
import os

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """Screen2words dataset."""
_CITATION = """
@inproceedings{wang2021screen2words,
    title={Screen2words: Automatic mobile UI summarization with multimodal
           learning},
    author={Wang, Bryan and
            Li, Gang and
            Zhou, Xin and
            Chen, Zhourong and
            Grossman, Tovi and
            Li, Yang},
    booktitle={The 34th Annual ACM Symposium on User Interface Software
               and Technology},
    pages={498--510},
    year={2021}
}
"""

# When running locally (recommended), copy files as above an use these:
_SCREEN2WORDS_DIR = "/tmp/data/rico/screen2words"
_RICO_DIR = "/tmp/data/rico/combined"


# (name, path) tuples for splits to be generated.
_SPLITS_TO_GENERATE = ["train", "dev", "test"]


class Screen2Words(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the Screen2words dataset."""

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {"1.0.0": "First release."}

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image/id": tfds.features.Scalar(np.int32),
            "image/filename": tfds.features.Text(),
            "image": tfds.features.Image(encoding_format="jpeg"),
            "summary": tfds.features.Sequence(tfds.features.Text()),
        }),
        supervised_keys=None,
        homepage="https://github.com/google-research-datasets/screen2words",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split)
            for split in _SPLITS_TO_GENERATE}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples from test set."""
    id_list_fname = os.path.join(
        _SCREEN2WORDS_DIR, "split", f"{split}_screens.txt")
    with open(id_list_fname, "r") as fin:
      split_ids = fin.readlines()

    summaries_fname = os.path.join(_SCREEN2WORDS_DIR, "screen_summaries.csv")
    summaries = collections.defaultdict(list)
    with open(summaries_fname, "r") as fin:
      for entry in csv.DictReader(fin):
        summaries[int(entry["screenId"])].append(entry["summary"])

    for line in split_ids:
      line = line.strip()
      image_id = int(line)
      yield image_id, {
          "image/id": image_id,
          "image/filename": f"{image_id}.jpg",
          "image": os.path.join(_RICO_DIR, f"{image_id}.jpg"),
          "summary": summaries[image_id],
      }
