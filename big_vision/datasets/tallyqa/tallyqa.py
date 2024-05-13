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

"""Import TallyQA into TFDS format. Uses Visual Genome and COCO images.

It's small data, so simple to run locally. First, download all the data:

  mkdir /tmp/data/ ; cd /tmp/data
  wget http://images.cocodataset.org/zips/{train2014,val2014}.zip
  wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
  wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
  wget https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true
  unzip *.zip

Then, update the PATHs below and run conversion locally like so (make sure to
install tensorflow-datasets for the `tfds` util):

  cd big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=tallyqa

Example to load:
  import tensorflow_datasets as tfds
  dataset = tfds.load('tallyqa', split='train', data_dir='/tmp/tfds')

The test split distinguishes between simple and complex questions. The train
split does not contain this information. We therefore set issimple to `-1` in
the train split to indicate it is not known.
"""

import json

import numpy as np
import tensorflow_datasets as tfds


_TALLYQA_PATH = '/tmp/data/tallyQA/'
_VISUAL_GENOME_PATH = '/tmp/data/visual_genome/'

_COCO_PATH = '/tmp/data/coco/'


_DESCRIPTION = """
TallyQA: Answering Complex Counting Questions
Most counting questions in visual question answering (VQA) datasets are simple
and require no more than object detection. Here, we study algorithms for complex
counting questions that involve relationships between objects, attribute
identification, reasoning, and more. To do this, we created TallyQA, the world's
largest dataset for open-ended counting.
"""

_CITATION = """
@inproceedings{acharya2019tallyqa,
  title={TallyQA: Answering Complex Counting Questions},
  author={Acharya, Manoj and Kafle, Kushal and Kanan, Christopher},
  booktitle={AAAI},
  year={2019}
}
"""

_HOMEPAGE = 'https://github.com/manoja328/TallyQA_dataset'


class TallyQA(tfds.core.GeneratorBasedBuilder):
  """Import TallyQA dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  There are three parts which should be downloaded:
  * TallyQA (train / test json files)
  * Visual Genome images (needed for train and test split)
  * COCO (2014) train / val images (only needed for train split)
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(None, None, 3)),
        'image_id': tfds.features.Scalar(dtype=np.int32),
        'image_source': tfds.features.Text(),
        'question': tfds.features.Text(),
        'question_id': tfds.features.Scalar(dtype=np.int32),
        'answer': tfds.features.Scalar(dtype=np.int32),
        'issimple': tfds.features.Scalar(dtype=np.int32),
    })

    return tfds.core.DatasetInfo(
        builder=self,
        features=features,
        description=_DESCRIPTION,
        supervised_keys=None,
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager) -> ...:
    """Call the function which defines the splits."""
    del dl_manager
    return {
        'train': self._generate_examples(split='train'),
        'test': self._generate_examples(split='test'),
    }

  def _generate_examples(self, split: str) -> ...:
    tally_json_file = f'{_TALLYQA_PATH}/{split}.json'
    with open(tally_json_file, 'r') as f:
      tally_json = json.load(f)

    for tally_qa in tally_json:
      # The TallyQA images come from two sources: Visual Genome and COCO.
      # Determine the correct dataset by inspecting the prefix.
      filepath = tally_qa['image']
      if filepath.startswith('VG_100K'):
        filepath = _VISUAL_GENOME_PATH + filepath
      elif filepath.startswith('train2014') or filepath.startswith('val2014'):
        filepath = _COCO_PATH + filepath
      else:
        raise ValueError(f'Unknown image path: {filepath}')

      tally_qa_dict = {
          'image': filepath,
          'image_id': tally_qa['image_id'],
          'image_source': tally_qa['data_source'],
          'question': tally_qa['question'],
          'question_id': tally_qa['question_id'],
          'answer': int(tally_qa['answer']),
      }
      if split == 'test':
        # Field only present in test split.
        tally_qa_dict.update({'issimple': tally_qa['issimple']})
      else:
        # In the train split, we set issimple to -1 to indicate it is not known.
        tally_qa_dict.update({'issimple': -1})
      tally_qa_id = f'{tally_qa_dict["image_id"]} / {tally_qa_dict["question_id"]}'  # pylint: disable=line-too-long
      yield tally_qa_id, tally_qa_dict
