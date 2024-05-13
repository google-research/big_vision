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
r"""Unbatch RefCOCO, RefCOCO+, RefCOCOg datasets in TFDS structure."""

# Based on tensorflow_datasets/datasets/ref_coco

import io
import os
import pickle

import numpy as np
import PIL.Image
import pycocotools.coco
import tensorflow_datasets as tfds

_ROOT_PATH = '/tmp/data/'


class RefCocoConfig(tfds.core.BuilderConfig):
  """Config to specify each RefCoco variant."""

  def __init__(self, dataset, dataset_partition, **kwargs):
    name = f'{dataset}_{dataset_partition}'
    super(RefCocoConfig, self).__init__(name=name, **kwargs)
    self.dataset = dataset
    self.dataset_partition = dataset_partition


_DESCRIPTION = """RefCOCO, RefCOCO+, RefCOCOg datasets.

Images, boxes and segmentations are from the original COCO dataset
(Lin et al, ECCV 2014). The referential segmentations are from two different
sources:

1) RefCOCOg (Mao et al, CVPR 2016):
  - https://github.com/mjhucla/Google_Refexp_toolbox
  - This is the split used in the "refcocog_google" dataset. Note that this
    split has overlapping images in train/validation. The same split is also
    provided in 2).

2) Source of RefCOCO and RefCOCO+ (Yu et al, ECCV 2016):
  - https://github.com/lichengunc/refer
  - Apache License 2.0
  - Provides all the splits used for generation of these datasets, including the
    "refcocog_google" split that is identical with the split from 1).

For convenience, we provide an additional dataset "refcocox_combined" that
combines the datasets "refcoco_unc", "refcocoplus_unc", and "refcocog_umd",
unifying "testA" and "testB" into a single "test" split, and removing any images
from "train" that appear either in "validation" or "test".

Also for convenience, every split is unrolled twice (at the "objects" level and
at the "object/refs" level) and saved as "{split}_flat".
"""

# pylint: disable=line-too-long
_CITATION = r"""
@inproceedings{DBLP:conf/cvpr/MaoHTCY016,
  author       = {Junhua Mao and
                  Jonathan Huang and
                  Alexander Toshev and
                  Oana Camburu and
                  Alan L. Yuille and
                  Kevin Murphy},
  title        = {Generation and Comprehension of Unambiguous Object Descriptions},
  booktitle    = {2016 {IEEE} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2016, Las Vegas, NV, USA, June 27-30, 2016},
  pages        = {11--20},
  publisher    = {{IEEE} Computer Society},
  year         = {2016},
  url          = {https://doi.org/10.1109/CVPR.2016.9},
  doi          = {10.1109/CVPR.2016.9},
  timestamp    = {Fri, 24 Mar 2023 00:02:52 +0100},
  biburl       = {https://dblp.org/rec/conf/cvpr/MaoHTCY016.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{DBLP:conf/eccv/YuPYBB16,
  author       = {Licheng Yu and
                  Patrick Poirson and
                  Shan Yang and
                  Alexander C. Berg and
                  Tamara L. Berg},
  editor       = {Bastian Leibe and
                  Jiri Matas and
                  Nicu Sebe and
                  Max Welling},
  title        = {Modeling Context in Referring Expressions},
  booktitle    = {Computer Vision - {ECCV} 2016 - 14th European Conference, Amsterdam,
                  The Netherlands, October 11-14, 2016, Proceedings, Part {II}},
  series       = {Lecture Notes in Computer Science},
  volume       = {9906},
  pages        = {69--85},
  publisher    = {Springer},
  year         = {2016},
  url          = {https://doi.org/10.1007/978-3-319-46475-6\_5},
  doi          = {10.1007/978-3-319-46475-6\_5},
  timestamp    = {Wed, 07 Dec 2022 23:10:23 +0100},
  biburl       = {https://dblp.org/rec/conf/eccv/YuPYBB16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

# coco_data = json.load(open('annotations/instances_train2017.json'))
# [l['name'] for l in coco_data['licenses']]
LICENSES = [
    'Attribution-NonCommercial-ShareAlike License',
    'Attribution-NonCommercial License',
    'Attribution-NonCommercial-NoDerivs License',
    'Attribution License',
    'Attribution-ShareAlike License',
    'Attribution-NoDerivs License',
    'No known copyright restrictions',
    'United States Government Work',
]
# _licenses_map = {l['id']: i for i, l in enumerate(coco_data['licenses'])}
_licenses_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}

# pyformat: disable
# [c['name'] for c in coco_data['categories']]
CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]
# sorted(set(c['supercategory'] for c in coco_data['categories']))
SUPERCATEGORIES = [
    'accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture',
    'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle',
]
# pyformat: enable


# Will be exported into directory `$TFDS_DATA_DIR/ref_coco_bv`
# If the class name was `RefCOCO` then it would be exported into
# `$TFDS_DATA_DIR/ref_coco`, which would collide with the default TFDS dataset
# also named `ref_coco` (which has precedence over `data_dir` builder arg).
class RefCocoBv(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for RefCoco datasets."""

  VERSION = tfds.core.Version('1.4.0')
  RELEASE_NOTES = {
      '1.4.0': 'Added flat versions of all dataset splits.',
      '1.3.0': 'Added "refcocox_combined" dataset.',
      '1.2.0': 'Added "train_flat" splits.',
      '1.1.0': 'Added more features (mask etc), nested "refs" in "objects".',
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  1. Install https://pypi.org/project/pycocotools/.

  2. Download data (requires ~20G for COCO images):

        (mkdir -p /tmp/tfds/downloads/manual &&
        cd /tmp/tfds/downloads/manual &&
        wget http://images.cocodataset.org/zips/train2017.zip &&
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip &&
        wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip &&
        wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip &&
        wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip &&
        for zip in *.zip; do unzip $zip; done
        )

  3. Run the generation script with `TFDS_DATA_DIR=/tmp/tfds`
  """

  BUILDER_CONFIGS = [
      RefCocoConfig(dataset='refcoco', dataset_partition='unc'),
      RefCocoConfig(dataset='refcoco', dataset_partition='google'),
      RefCocoConfig(dataset='refcocoplus', dataset_partition='unc'),
      RefCocoConfig(dataset='refcocog', dataset_partition='google'),
      RefCocoConfig(dataset='refcocog', dataset_partition='umd'),
      RefCocoConfig(dataset='refcocox', dataset_partition='combined'),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'id': tfds.features.Scalar(np.int32),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'height': tfds.features.Scalar(np.int32),
            'width': tfds.features.Scalar(np.int32),
            'license': tfds.features.ClassLabel(names=LICENSES),
            'file_name': tfds.features.Text(),
            'flickr_url': tfds.features.Text(),
            'coco_url': tfds.features.Text(),
            'objects': tfds.features.Sequence({
                'id': tfds.features.Scalar(np.int64),
                'area': tfds.features.Scalar(np.float32),
                'bbox': tfds.features.BBoxFeature(),
                'mask': tfds.features.Image(encoding_format='png'),
                'category': tfds.features.ClassLabel(names=CATEGORIES),
                'supercategory': tfds.features.ClassLabel(
                    names=SUPERCATEGORIES
                ),
                'iscrowd': tfds.features.Scalar(np.bool_),
                # refcoco, refcoco+, refcocog features:
                'refs': tfds.features.Sequence({
                    'id': tfds.features.Scalar(np.int32),
                    'sentence': tfds.features.Text(),
                }),
            }),
        }),
        supervised_keys=None,  # Set to `None` to disable
        citation=_CITATION,
        description=_DESCRIPTION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    allowed_splits = {
        ('refcoco', 'google'): [
            tfds.Split.TRAIN,
            tfds.Split.VALIDATION,
            tfds.Split.TEST,
        ],
        ('refcoco', 'unc'): [
            tfds.Split.TRAIN,
            tfds.Split.VALIDATION,
            'testA',
            'testB',
        ],
        ('refcocoplus', 'unc'): [
            tfds.Split.TRAIN,
            tfds.Split.VALIDATION,
            'testA',
            'testB',
        ],
        # Verified manually that image and annotation IDs match the ones in
        # https://storage.googleapis.com/refexp/google_refexp_dataset_release.zip
        ('refcocog', 'google'): [
            tfds.Split.TRAIN,
            tfds.Split.VALIDATION,
        ],
        ('refcocog', 'umd'): [
            tfds.Split.TRAIN,
            tfds.Split.VALIDATION,
            tfds.Split.TEST,
        ],
        ('refcocox', 'combined'): [
            tfds.Split.TRAIN,
            tfds.Split.VALIDATION,
            tfds.Split.TEST,
        ],
    }
    bc = self.builder_config
    splits = allowed_splits[(bc.dataset, bc.dataset_partition)]

    data_dir = dl_manager.manual_dir
    for url, components in (
        # pylint: disable=line-too-long
        # pyformat: disable
        ('http://images.cocodataset.org/zips/train2017.zip', ('train2017', '000000147328.jpg')),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', ('annotations', 'instances_train2017.json')),
        ('https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip', ('refcoco', 'refs(unc).p')),
        ('https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip', ('refcoco+', 'refs(unc).p')),
        ('https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip', ('refcocog', 'refs(umd).p')),
        # pyformat: enable
        # pylint: enable=line-too-long
    ):
      path = os.path.exists(os.path.join(data_dir, *components))
      if not path:
        raise FileNotFoundError(
            f'Could not find {path}: please download {url} and unzip into'
            f' {data_dir}'
        )

    coco = pycocotools.coco.COCO(
        os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    )

    return {
        split + suffix: self._generate_examples(
            coco, data_dir, bc.dataset, bc.dataset_partition, split + suffix,
        )
        for split in splits
        for suffix in ('', '_flat')
    }

  # Builder must overwrite all abstract methods.
  def _generate_examples(
      self, coco, data_dir, dataset, dataset_partition, split):
    return _generate_examples(coco, data_dir, dataset, dataset_partition, split)


def _get_ids(data_dir, dataset, dataset_partition, split):
  """Returns `img_ids, ann_to_refs` for specified dataset/partition/split."""

  def load(dataset, dataset_partition):
    fname = f'refs({dataset_partition}).p'
    path = os.path.join(data_dir, dataset, fname)
    refcoco = pickle.load(open(path, 'rb'))
    return refcoco

  if split == tfds.Split.VALIDATION:
    split = 'val'

  if (dataset, dataset_partition) == ('refcocox', 'combined'):
    refcoco = (
        load('refcocog', 'umd')
        + load('refcoco', 'unc')
        + load('refcoco+', 'unc')
    )
    if split == 'test':
      splits = ('test', 'testA', 'testB')
    else:
      splits = (split,)

    exclude_img_ids = set()
    if split == 'train':
      # Exclude all images with val/test annotations from train set.
      exclude_img_ids = {
          r['image_id'] for r in refcoco if r['split'] != 'train'
      }
    refcoco = [
        r
        for r in refcoco
        if r['split'] in splits and r['image_id'] not in exclude_img_ids
    ]

  else:
    if dataset == 'refcocoplus':
      dataset = 'refcoco+'
    refcoco = load(dataset, dataset_partition)
    refcoco = [r for r in refcoco if r['split'] == split]

  img_ids = {r['image_id'] for r in refcoco}
  ann_to_refs = {}
  for r in refcoco:
    for sent in r['sentences']:
      ann_to_refs.setdefault(r['ann_id'], []).append(dict(
          id=sent['sent_id'],
          sentence=sent['sent']
      ))

  return img_ids, ann_to_refs


def _generate_examples(coco, data_dir, dataset, dataset_partition, split):
  """Generates examples for a given split."""

  flat = '_flat' in split
  split = split.replace('_flat', '')
  img_ids, ann_to_refs = _get_ids(data_dir, dataset, dataset_partition, split)

  for img_id in coco.getImgIds():

    if img_id not in img_ids:
      continue
    img, = coco.loadImgs([img_id])

    example = {
        'id': img_id,
        'image': os.path.join(data_dir, 'train2017', img['file_name']),
        'height': img['height'],
        'width': img['width'],
        'license': LICENSES[_licenses_map[img['license']]],
        'file_name': img['file_name'],
        'flickr_url': img['flickr_url'],
        'coco_url': img['coco_url'],
        'objects': [],
    }
    for ann in coco.loadAnns(coco.getAnnIds(img_id)):
      refs = ann_to_refs.get(ann['id'])
      if not refs:
        continue
      cat, = coco.loadCats([ann['category_id']])
      mask = coco.annToMask(ann).astype(np.bool_)
      mask_buf = io.BytesIO()
      PIL.Image.fromarray(mask).save(mask_buf, 'png')
      mask_buf.seek(0)
      object_ = {
          'id': ann['id'],
          'mask': mask_buf,
          'category': cat['name'],
          'supercategory': cat['supercategory'],
          'iscrowd': ann['iscrowd'],
          'area': ann['area'],
          'bbox': _convert_bbox(img, *ann['bbox']),
          'refs': refs,
      }
      if flat:
        example['objects'] = [object_]
        for ref_i, ref in enumerate(refs):
          object_['refs'] = [ref]
          mask_buf.seek(0)
          yield f'{img_id}_{ann["id"]}_{ref_i}', example
      else:
        example['objects'].append(object_)

    if not flat:
      yield img_id, example


def _convert_bbox(img, x, y, w, h):
  return tfds.features.BBox(
      ymin=y / img['height'],
      xmin=x / img['width'],
      ymax=(y + h) / img['height'],
      xmax=(x + w) / img['width'],
  )
