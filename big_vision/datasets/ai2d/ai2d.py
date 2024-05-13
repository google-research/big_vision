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
r"""AI2D TFDS converter.


It's a small dataset, so can be built locally. Copy the data to local disk:

  mkdir -p /tmp/data/ai2d
  unzip -d /tmp Downloads/ai2d-all.zip
  mv Downloads/ai2d_test_ids.csv /tmp/ai2d/

Also download a font for rendering, set the location in the flag font_path.

Then, run conversion locally (make sure to install tensorflow-datasets for the `tfds` util):

  cd third_party/py/big_vision/datasets
  env TFDS_DATA_DIR=/tmp/tfds tfds build --datasets=ai2d

Example to load:

  import tensorflow_datasets as tfds
  dataset = tfds.load(ai2d', split='train', data_dir='/tmp/tfds')
"""

import functools
import glob
import io
import json
import os
from typing import Any, Dict

from absl import flags
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow_datasets as tfds


_DESCRIPTION = """AI2D dataset."""

# pylint: disable=line-too-long
_CITATION = """
@inproceedings{kembhavi2016eccv,
  author = {Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi},
  title = {A Diagram Is Worth A Dozen Images},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2016}
  url={https://api.semanticscholar.org/CorpusID:2682274}
}
"""
# pylint: enable=line-too-long


_INPUT_PATH = flags.DEFINE_string(
    'input_path', '/tmp/ai2d/', 'Downloaded AI2D data.'
)
_FONT_PATH = flags.DEFINE_string(
    'font_path', '/tmp/DMSans-Regular.ttf', 'Font for rendering annotations.'
)


class Ai2d(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for AI2D dataset."""

  VERSION = tfds.core.Version('1.1.0')
  RELEASE_NOTES = {'1.1.0': 'Re-create from scratch + more fields.'}

  def _info(self):
    """Returns the metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'id': tfds.features.Text(),
            'question': tfds.features.Text(),
            'label': tfds.features.Scalar(np.int32),
            'answer': tfds.features.Text(),
            'possible_answers': tfds.features.Sequence(tfds.features.Text()),
            'abc_label': tfds.features.Scalar(np.bool_),
            'image_name': tfds.features.Text(),
            'image': tfds.features.Image(encoding_format='png'),
        }),
        homepage='https://allenai.org/data/diagrams',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {split: self._generate_examples(split)
            for split in ('test', 'train')}

  def _generate_examples(self, split: str):
    """Yields (key, example) tuples."""
    with open(
        os.path.join(_INPUT_PATH.value, 'ai2d_test_ids.csv'), 'r'
    ) as f:
      all_test_ids = f.readlines()
    all_test_ids = [line.strip() for line in all_test_ids]

    all_annotation_paths = glob.glob(
        os.path.join(_INPUT_PATH.value, 'questions', '*.json'))
    for annotation_path in all_annotation_paths:
      basename = os.path.basename(annotation_path)
      image_id = basename.split('.')[0]
      if image_id in all_test_ids and split == 'train':
        continue
      elif image_id not in all_test_ids and split == 'test':
        continue

      text_annotation_path = os.path.join(
          _INPUT_PATH.value, 'annotations', basename
      )
      with open(annotation_path, 'r') as f:
        with open(text_annotation_path, 'r') as g:
          question_json = json.load(f)
          text_annotation_json = json.load(g)
          for question in question_json['questions']:
            label_id = int(
                question_json['questions'][question]['correctAnswer']
            )
            choices = question_json['questions'][question]['answerTexts']
            abc_label = question_json['questions'][question]['abcLabel']
            annotation = {
                'id': question_json['questions'][question]['questionId'],
                'question': question,
                'label': label_id,
                'answer': choices[label_id],
                'possible_answers': tuple(choices),
                'abc_label': abc_label,
                'image_name': question_json['imageName'],
            }
            annotation['image'] = _create_image(
                annotation, text_annotation_json['text']
            )
            yield annotation['id'], annotation


@functools.cache
def Font(  # pylint: disable=invalid-name
    size: int,
) -> ImageFont.FreeTypeFont:
  """Loads the font from in the specified style.

  Args:
    size: The size of the returned font.

  Returns:
    The loaded font.
  """
  return ImageFont.truetype(_FONT_PATH.value, size=size)


def _create_image(
    annotation: Dict[str, Any], text_annotation: Dict[str, Any]
) -> bytes:
  """Adds image to one annotation."""
  img_path = os.path.join(_INPUT_PATH.value, 'images', annotation['image_name'])
  with open(img_path, 'rb') as f:
    if annotation['abc_label']:
      raw_image = _draw_text(f, text_annotation)
    else:
      raw_image = f.read()
  return raw_image


def _draw_text(image: bytes, text_annotations: Dict[str, Any]) -> bytes:
  """Replaces text in image by the correct replacement letter from AI2D."""
  image = Image.open(image)
  draw = ImageDraw.Draw(image)
  for annotation in text_annotations:
    current_annotation = text_annotations[annotation]
    rectangle = current_annotation['rectangle']
    box = [tuple(rectangle[0]), tuple(rectangle[1]),]
    text = current_annotation['replacementText']
    position = box[0]
    draw.rectangle(box, fill='white')
    font_size = 100
    x_diff = box[1][0] - box[0][0]
    y_diff = box[1][1] - box[0][1]
    font = Font(font_size)
    size = font.getbbox(text)
    while (size[2] > x_diff or size[3] > y_diff) and font_size > 0:
      font = Font(font_size)
      size = font.getbbox(text)
      font_size -= 1
    delta = (x_diff - size[2]) // 2
    position = (position[0] + delta, position[1])
    draw.text(position, text, fill='black', font=font)
  new_image_bytes = io.BytesIO()
  image.save(new_image_bytes, format='PNG')
  return new_image_bytes.getvalue()
