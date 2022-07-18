# Copyright 2022 Big Vision Authors.
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

"""Multi-host image->text and text->image retrieval evaluation.

Example how to add to config:

  config.evals {}
  config.evals.retieval = dict(log_steps=1200, type='proj.image_text.retrieval')
  config.evals.retrieval.dataset = 'coco_captions'
  config.evals.retrieval.txt_name = ('captions', 'text')
  # Note that initial "decode|" is not needed.
  config.evals.retrieval.pp_img = 'resize(224)|value_range(-1,1)'
  # Raw text strings use key "texts" in feature dict. The evaluator expects
  # tokenized text with key "labels".
  config.evals.retrieval.pp_txt = (
      'tokenize(max_len=16, eos="sticky", pad_value=1, inkey="texts", '
      '         outkey="labels")')

Example to support precomputed data:
See `big_vision/configs/proj/image_text/lit.py`.
"""

import functools
import operator
import time

from absl import logging
from big_vision.evaluators.proj.image_text import image_text_retrieval
import big_vision.pp.builder as pp_builder
from clu import deterministic_data
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _with_infinite_padding(dataset):
  """Adds "infinite padding" to the dataset."""
  filler_element = tf.nest.map_structure(
      lambda spec: tf.zeros(spec.shape, spec.dtype)[None], dataset.element_spec)
  filler_element["mask"] = [False]
  filler_dataset = tf.data.Dataset.from_tensor_slices(filler_element)
  dataset = dataset.map(
      lambda features: dict(mask=True, **features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset.concatenate(filler_dataset.repeat(None))


def _pad_and_batch(dataset, batch_dims):
  """Adds padding and then batches dataset."""
  dataset = _with_infinite_padding(dataset)
  for batch_size in reversed(batch_dims):
    dataset = dataset.batch(batch_size)
  return dataset


# This is needed so retrieval_test can replace dataset info.
def _get_dataset_info(builder):
  return builder.info


def prepare_datasets(dataset, *, pp_img, pp_txt, txt_name, offset=0):
  """Returns unbatched `ds_images, ds_texts` datasets.

  Args:
    dataset: An image-text `tf.data.Dataset` that is expected to contain the
      following features: "image" (dtype=uint8, shape=[None, None, 3]),
      `txt_name` (dtype=string, shape=[None]).
    pp_img: String defining pre-processing for images. The pre-processing can
      expect the following features to be prepared: "image", "id". The
      pre-processing should convert the "image" (dtype=uint8,
      shape=[None, None, 3]) to "image" (dtype=float32, shape=[sz, sz, 3]).
    pp_txt: String defining pre-processing for text. The pre-processing can
      expect the following features to be prepared: "texts", "id", "caption_id".
      The pre-processing should convert the "texts" (dtype=string, shape=[])
      into a tokenized "labels" (dtype=int32, shape=[max_len]).
    txt_name: Name of the text feature to unroll in the original `dataset`. Can
      be a simple string feature name, or an iterable of strings to specify a
      nested feature (e.g. for "coco_captions", this would be
      `('captions', 'text')`).
    offset: Offset that should be added to enumerated examples to generate IDs.
      In a multi-host setup, this is typically set to a value large enough to
      make all IDs distinct.

  Returns:
    Image and text datasets.
  """

  def get_feature_value(data, feature_name):
    if isinstance(feature_name, str):
      feature_name = [feature_name]
    return functools.reduce(operator.getitem, feature_name, data)

  def get_captions(idx, features):
    """Returns a dataset with unrolled "caption" for every example."""
    texts = get_feature_value(features, txt_name)
    texts_n = tf.shape(texts)[0]
    return tf.data.Dataset.from_tensor_slices({
        "id": tf.tile([idx + offset], [texts_n]),
        "caption_i": tf.stack(tf.range(texts_n)),
        "texts": tf.stack(texts),
    })

  def add_id(idx, features):
    return {**features, "id": idx + offset}

  ds_images = dataset.enumerate().map(add_id).map(
      pp_builder.get_preprocess_fn(
          f"{pp_img}|keep('id', 'image')", remove_tpu_dtypes=False))
  ds_texts = dataset.enumerate().flat_map(get_captions).map(
      pp_builder.get_preprocess_fn(
          f"{pp_txt}|keep('id', 'caption_i', 'labels')",
          remove_tpu_dtypes=False))
  return ds_images, ds_texts


def _split_and_batch(dataset_name, batch_size, split, get_ds, data_dir=None):
  """Splits dataset, calls `get_ds` and returns padded + batched datasets."""
  assert not batch_size % jax.device_count(), (
      f"batch_size={batch_size} % jax.device_count()={jax.device_count()}")
  builder = tfds.builder(dataset_name, data_dir=data_dir)
  batch_dims = [
      jax.local_device_count(), batch_size // jax.device_count()
  ]
  info = _get_dataset_info(builder)
  num_examples = info.splits[split].num_examples
  read_instruction = deterministic_data.get_read_instruction_for_host(
      split=split,
      dataset_info=info,
      remainder_options=deterministic_data.RemainderOptions.ON_FIRST_PROCESS)
  ds_images, ds_texts = get_ds(builder.as_dataset(split=read_instruction),
                               offset=jax.process_index() * num_examples)
  return (
      _pad_and_batch(ds_images, batch_dims),
      _pad_and_batch(ds_texts, batch_dims),
  )


class Evaluator:
  """Image/text retrieval evaluator."""

  def __init__(self,
               predict_fn,
               *,
               dataset,
               pp_img,
               pp_txt,
               txt_name,
               batch_size,
               data_dir=None,
               split="test"):
    """Initializes a new zero-shot image/text retrieval evaluator.

    See `prepare_datasets()` for details on how the dataset is pre-processed.

    Args:
      predict_fn: Prediction function with signature
        `zimg, ztxt, out = predict_fn(params, images, texts)`
      dataset: The TFDS dataset name of the eval data.
      pp_img: Preprocessing string for images. Preprocessed features should
        contain key "image" with value that can be batched and is suitable for
        `predict_fn(images)` input``.
      pp_txt: Preprocessing string for texts. Can expect "texts" key as an input
        (shape=[], dtype=string), and is expected to produce "labels" key that
        is suitable for `predict_fn(texts)` input.
      txt_name: The name of the feature of captions (can be a tuple to look up a
        value in a nested feature dictionary). Expected shape=[None],
        dtype=string. specified then items are used as lookup path.
      batch_size: Global batch size.
      data_dir: Optional dir to load the TFDS dataset from.
      split: The split of the eval data.
    """
    self.ds_images, self.ds_texts = _split_and_batch(
        dataset, batch_size, split,
        functools.partial(
            prepare_datasets, pp_img=pp_img, pp_txt=pp_txt, txt_name=txt_name),
        data_dir=data_dir)
    self._axis_name = "batch"

    def embed_images(params, images):
      zimg, _, _ = predict_fn(params, images, None)
      return jax.lax.all_gather(zimg, axis_name=self._axis_name)

    def embed_texts(params, texts):
      _, ztxt, _ = predict_fn(params, None, texts)
      return jax.lax.all_gather(ztxt, axis_name=self._axis_name)

    self._embed_images_p = jax.pmap(embed_images, axis_name=self._axis_name)
    self._embed_texts_p = jax.pmap(embed_texts, axis_name=self._axis_name)
    self._all_gather_p = jax.pmap(
        lambda x: jax.lax.all_gather(x, axis_name=self._axis_name),
        axis_name=self._axis_name)
    self._count_p = jax.pmap(
        lambda mask: jax.lax.psum(mask.sum(), axis_name=self._axis_name),
        axis_name=self._axis_name)
    self._compiled = set()

  def _embed(self, name, params, ds, embed_fn, id_names):
    """Embeds features name `name` using `embed_fn`.

    Args:
      name: Feature name to be embedded.
      params: Parameters for the predict_fn.
      ds: The dataset.
      embed_fn: A pmapped function that returns the embeddings.
      id_names: An iterable of feature names that should be collected.

    Returns:
      A dictionary with "embeddings" and `id_names` as keys.
    """
    ns = []
    embeddings = []
    ids = {id_name: [] for id_name in list(id_names) + ["mask"]}

    t0 = time.time()

    for batch in ds:
      ns.append(self._count_p(np.asarray(memoryview(batch["mask"])))[0])
      # Due to infinite padding, this loop will never end. We will stop once
      # all processes only process padded data. We don't check the latest
      # DeviceArray `ns[-1]` Because we want to keep our computation async for
      # efficiency reasons.
      if len(ns) >= 2 and ns[-2] == 0:
        break

      embs = embed_fn(params, np.asarray(memoryview(batch[name])))[0]
      if embed_fn not in self._compiled:
        logging.info("Compiled %s embeddings in %.3fs", name, time.time() - t0)
        t0 = time.time()
        self._compiled.add(embed_fn)

      embeddings.append(embs.reshape([-1, embs.shape[-1]]))
      for id_name in ids:
        ids[id_name].append(
            self._all_gather_p(np.array(batch[id_name]))[0].flatten())

    # Only access DeviceArray at end of loop for better efficiency.
    ns = np.array(ns)
    embeddings = np.concatenate(embeddings)
    ids = {k: np.concatenate(v) for k, v in ids.items()}
    masks = ids.pop("mask").astype(bool)
    logging.info("Processed %s in %d steps - ...%s", name, len(ns), ns[-10:])
    n = ns.sum()
    logging.info("Totalling %d %s in %.3fs", n, name, time.time() - t0)
    return {
        "embeddings": embeddings[masks],
        **{k: v[masks] for k, v in ids.items()},
    }

  def evaluate(self, params):
    """Returns evaluation results."""
    images = self._embed("image", params, self.ds_images, self._embed_images_p,
                         ("id",))
    texts = self._embed("labels", params, self.ds_texts, self._embed_texts_p,
                        ("id", "caption_i"))
    # Shapes: (nimg, emb) * (emb, ntxt) -> (nimg, ntxt)
    similarities = np.dot(images["embeddings"], texts["embeddings"].T)

    t0 = time.time()
    id2img = {id_: i for i, id_ in enumerate(images["id"])}
    text_image_correspondence = [id2img[id_] for id_ in texts["id"]]
    img2txt = image_text_retrieval.image_to_text_retrieval_eval(
        -similarities, text_image_correspondence)
    txt2img = image_text_retrieval.text_to_image_retrieval_eval(
        -similarities, text_image_correspondence)
    logging.info("Computed retrieval metrics in %.3fs", time.time() - t0)

    return dict(
        images=images,
        texts=texts,
        img2txt=img2txt,
        txt2img=txt2img,
    )

  def run(self, params):
    """Returns metrics."""
    results = self.evaluate(params)
    return [(f"{direction}_{k.lower()}", v)
            for direction in ("img2txt", "txt2img")
            for k, v in results[direction].items()]
