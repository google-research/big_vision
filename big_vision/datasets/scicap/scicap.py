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
r"""Creates TFDS dataset for SciCap.

Preparing the data:
  1) mkdir /tmp/data/scicap && cd /tmp/data/scicap
  2) wget 'https://www.dropbox.com/s/t1sjqesl0pynaxo/scicap_data.zip?dl=0'
  3) unzip -UU 'scicap_data.zip?dl=0' && rm 'scicap_data.zip?dl=0'

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=scicap

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load('scicap', split='train', data_dir='/tmp/tfds')
"""
# pylint: enable=line-too-long
import enum
import functools
import json
import os

import tensorflow_datasets as tfds


_DESCRIPTION = """SciCap dataset."""
_CITATION = """
@article{hsu2021scicap,
    title={SciCap: Generating captions for scientific figures},
    author={Hsu, Ting-Yao and Giles, C Lee and Huang, Ting-Hao'Kenneth'},
    journal={arXiv preprint arXiv:2110.11624},
    year={2021}
}
"""

# When running locally (recommended), copy files as above an use these:
_SCICAP_DIR = "/tmp/data/scicap/scicap_data"


class ScicapSubset(enum.Enum):
  """Versions of the SciCap dataset."""
  SINGLE_SENTENCE = "single_sentence"
  FIRST_SENTENCE = "first_sentence"
  LEQ_100_TOKENS = "leq_100_tokens"

_SPLITS_TO_GENERATE = ["train", "test", "val"]
_CONFIG_TO_IDS_PATH = {
    (ScicapSubset.SINGLE_SENTENCE, True): "Single-Sentence-Caption/Yes-Subfig",
    (ScicapSubset.SINGLE_SENTENCE, False): "Single-Sentence-Caption/No-Subfig",
    (ScicapSubset.FIRST_SENTENCE, True): "First-Sentence/Yes-Subfig",
    (ScicapSubset.FIRST_SENTENCE, False): "First-Sentence/No-Subfig",
    (ScicapSubset.LEQ_100_TOKENS, True):
        "Caption-No-More-Than-100-Tokens/Yes-Subfig",
    (ScicapSubset.LEQ_100_TOKENS, False):
        "Caption-No-More-Than-100-Tokens/No-Subfig",
}
_SUBFIG_TO_PATH = {
    True: "SciCap-Yes-Subfig-Img", False: "SciCap-No-Subfig-Img"
}


class ScicapConfig(tfds.core.BuilderConfig):
  """"Configuration for SciCap caption length and subfigure inclusion."""

  def __init__(self, *, subset: ScicapSubset, subfig: bool, **kwargs):
    """Parameters specifying how the dataset will be processed.

    Args:
      subset: Subset of the Scicap data (see enum above).
      subfig: Whether or not figure with subfigures are included.
      **kwargs: Passed on to the constructor of `BuilderConfig`.
    """
    super(ScicapConfig, self).__init__(**kwargs)
    self.subset = subset
    self.subfig = subfig


@functools.cache
def _read_annotations(split: str, image_id: str):
  """Reads annotations for a single file."""
  path = os.path.join(_SCICAP_DIR, "SciCap-Caption-All", split)
  fname = os.path.join(path, image_id + ".json")
  with open(fname, "r") as fin:
    return json.load(fin)


class Scicap(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the SciCap dataset."""

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {"1.0.0": "First release."}

  BUILDER_CONFIGS = [
      ScicapConfig(
          name="single_sentence_subfig_yes",
          description="Single sentence caption with subfigures allowed.",
          subset=ScicapSubset.SINGLE_SENTENCE,
          subfig=True
      ),
      ScicapConfig(
          name="single_sentence_subfig_no",
          description="Single sentence caption with subfigures not allowed.",
          subset=ScicapSubset.SINGLE_SENTENCE,
          subfig=False
      ),
      ScicapConfig(
          name="first_sentence_subfig_yes",
          description="First sentence of captions with subfigures allowed.",
          subset=ScicapSubset.FIRST_SENTENCE,
          subfig=True
      ),
      ScicapConfig(
          name="first_sentence_subfig_no",
          description="First sentence of captions with subfigures not allowed.",
          subset=ScicapSubset.FIRST_SENTENCE,
          subfig=False
      ),
      ScicapConfig(
          name="leq_100_tokens_subfig_yes",
          description="Captions with <= 100 tokens with subfigures allowed.",
          subset=ScicapSubset.LEQ_100_TOKENS,
          subfig=True
      ),
      ScicapConfig(
          name="leq_100_tokens_subfig_no",
          description=("Captions with <= 100 tokens with subfigures"
                       " not allowed."),
          subset=ScicapSubset.LEQ_100_TOKENS,
          subfig=False
      ),
  ]

  def _info(self):
    """Returns the metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image/id": tfds.features.Text(),
            "image/filename": tfds.features.Text(),
            "image": tfds.features.Image(encoding_format="png"),
            "caption/originally_extracted": tfds.features.Text(),
            "caption/lowercase_and_token_and_remove_figure_index":
                tfds.features.Text(),
            "caption/normalized/basic_num": tfds.features.Text(),
            "caption/normalized/advanced_equation_bracket":
                tfds.features.Text(),
        }),
        supervised_keys=None,
        homepage="https://github.com/tingyaohsu/SciCap",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split)
            for split in _SPLITS_TO_GENERATE}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples from test set."""
    config_path = _CONFIG_TO_IDS_PATH[
        (self.builder_config.subset, self.builder_config.subfig)]
    image_path = os.path.join(
        _SCICAP_DIR, _SUBFIG_TO_PATH[self.builder_config.subfig], split)
    id_list_fname = os.path.join(
        _SCICAP_DIR, "List-of-Files-for-Each-Experiments",
        config_path, split, "file_idx.json")
    with open(id_list_fname, "r") as fin:
      split_images = json.load(fin)

    for fname in split_images:
      assert fname.endswith(".png")
      image_id = fname[:-len(".png")]
      annotations = _read_annotations(split, image_id)
      yield fname, {
          "image/id": image_id,
          "image/filename": fname,
          "image": os.path.join(image_path, fname),
          "caption/originally_extracted": annotations["0-originally-extracted"],
          "caption/lowercase_and_token_and_remove_figure_index":
              annotations["1-lowercase-and-token-and-remove-figure-index"][
                  "caption"],
          "caption/normalized/basic_num": annotations["2-normalized"][
              "2-1-basic-num"]["caption"],
          "caption/normalized/advanced_equation_bracket":
              annotations["2-normalized"][
                  "2-2-advanced-euqation-bracket"]["caption"]
      }
