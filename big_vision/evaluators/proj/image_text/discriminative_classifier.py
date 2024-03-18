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

"""Discriminative zero-shot classification evaluator.
"""

import functools
import time

from absl import logging
from big_vision import input_pipeline
from big_vision import utils
from big_vision.evaluators.proj.image_text import prompt_engineering
from big_vision.pp import ops_general  # pylint: disable=unused-import
from big_vision.pp import ops_image  # pylint: disable=unused-import
import big_vision.pp.builder as pp_builder
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


DATASET_NAMES = ("imagenet2012", "cifar100", "oxford_iiit_pet")
DEFAULT_OVERRIDES = (
    ("imagenet2012", (
        ("class_names", "clip"),
        ("split", "validation"),
        )),
    )


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


# This is needed so retrieval_test can replace dataset info.
def _get_dataset_info(builder):
  return builder.info


def prepare_datasets(img_dataset,
                     class_names,
                     *,
                     prompt_templates,
                     pp_img,
                     pp_txt,
                     cache_final=False,
                     pre_filter_fn=None,
                     class_name_offset=0):
  """Returns unbatched `ds_images, ds_texts` datasets."""

  assert prompt_templates, "Must specify prompt templates (e.g. simply ['{}'])"

  def expand_aliases(idx, class_name):
    class_names = tf.strings.split(class_name, ",")
    return tf.data.Dataset.from_tensor_slices((
        tf.repeat([idx + class_name_offset], len(class_names), axis=0),
        class_names,
    ))

  def add_prompts(idx, class_name):
    return tf.data.Dataset.from_tensor_slices({
        "label": tf.repeat([idx], len(prompt_templates), axis=0),
        "class_name": tf.repeat([class_name], len(prompt_templates), axis=0),
        "prompt_template": prompt_templates,
    })

  def substitute_prompt(features):
    parts = tf.strings.split(features["prompt_template"], "{}")
    tf.debugging.assert_equal(len(parts), 2, features["prompt_template"])
    return {
        "label": features["label"],
        "texts": tf.strings.join([parts[0], features["class_name"], parts[1]])
    }

  if pre_filter_fn:
    img_dataset = img_dataset.filter(pre_filter_fn)
  ds_images = img_dataset.map(
      pp_builder.get_preprocess_fn(f"{pp_img}|keep('label', 'image')"))
  ds_texts = tf.data.Dataset.from_tensor_slices(list(class_names)).enumerate(
  ).flat_map(expand_aliases).flat_map(add_prompts).map(substitute_prompt).map(
      pp_builder.get_preprocess_fn(f"{pp_txt}|keep('label', 'labels')"))

  if cache_final:
    ds_images, ds_texts = ds_images.cache(), ds_texts.cache()

  return ds_images, ds_texts


def _split_and_batch(dataset_name, data_dir, class_names, batch_size, split,
                     get_ds):
  """Splits dataset, calls `get_ds` and returns padded + batched datasets."""
  assert not batch_size % jax.device_count(), (
      f"batch_size={batch_size} % jax.device_count()={jax.device_count()}")
  builder = tfds.builder(dataset_name, data_dir=data_dir)

  # Split class names (last process gets remainder).
  if len(class_names) < jax.process_count():
    # See (internal link) for more details.
    class_names += [""] * (jax.process_count() - len(class_names))
  per_process = len(class_names) // jax.process_count()
  class_name_offset = per_process * jax.process_index()
  if jax.process_index() == jax.process_count() - 1:
    class_names = class_names[class_name_offset:]
  else:
    class_names = class_names[class_name_offset:class_name_offset + per_process]

  ds_images, ds_texts = get_ds(
      builder.as_dataset(split=tfds.split_for_jax_process(split)),
      class_names,
      class_name_offset=class_name_offset)
  return (
      _with_infinite_padding(ds_images).batch(batch_size),
      _with_infinite_padding(ds_texts).batch(batch_size),
  )


def _average_embeddings(embeddings, *, labels, num_classes, normalize):
  """Computes per-class averages of `embeddings`."""
  assert embeddings.ndim == 2, f"Expected {embeddings.ndim}==2"
  assert labels.ndim == 1, f"Expected {labels.ndim}==1"
  assert len(labels) == len(embeddings), (
      f"Expected {len(labels)}=={len(embeddings)}")

  byidx = [[] for _ in range(num_classes)]
  for label, embedding in zip(labels, embeddings):
    byidx[label].append(embedding)
  missing = set(range(num_classes)) - set(
      idx for idx, embs in enumerate(byidx) if len(embs))
  assert not missing, f"Classes without embeddings: {missing}"
  embeddings = [np.array(embedding).mean(axis=0) for embedding in byidx]
  embeddings = np.stack(embeddings)

  assert len(embeddings) == num_classes
  if normalize:
    embeddings /= 1e-8 + np.linalg.norm(embeddings, axis=1, keepdims=True)
  return embeddings


class Evaluator:
  """Zero-shot classification evaluator."""

  def __init__(self,
               predict_fn,
               *,
               batch_size,
               devices,
               dataset_names=DATASET_NAMES,
               data_dir=None,
               class_names="dataset_info:label",
               split="test",
               prompt_templates="clip_paper",
               canonicalize=True,
               pp_img="resize(224)|value_range(-1,1)",
               pp_txt="tokenize(max_len=16, eos='sticky', "
                      "pad_value=1, inkey='texts', outkey='labels')",
               cache_final=False,
               pre_filter_fn=None,
               first_class_name_only=True,
               dataset_overrides=DEFAULT_OVERRIDES,
               async_delay=1):
    """Initializes a new zero-shot classification evaluator.

    See `prepare_datasets()` for details on how the dataset is pre-processed.

    Args:
      predict_fn: Prediction function with signature
        `zimg, ztxt, out = predict_fn(params, images, texts)`
      batch_size: Global batch size.
      devices: list of devices.
      dataset_names: Names of TFDS datasets to evaluate on.
      data_dir: Optional argument to `tfds.builder()`.
      class_names: Usually specified as a string that is interpreted by
        `prompt_engineering.get_class_names()` to look up class names.
        Alternatively, this attribute can be a list of class names (using ","
        to separate multiple aliases).
      split: Which dataset split to use for evaluation.
      prompt_templates: Specifies which prompt templates to use. See module
        big_vision.evaluators.proj.image_text.prompte_engineering
        for valid values.
      canonicalize: Whether class names and prompt templates should be
        canonicalized. See `prompt_engineering.py` for details.
      pp_img: Preprocessing string for images. Preprocessed features should
        contain key "image" with value that can be batched and is suitable for
        the `images` argument of `predict_fn` input``.
      pp_txt: Preprocessing string for texts. Can expect "texts" key as an input
        (shape=[], dtype=string), and is expected to produce "labels" key that
        is suitable for the `text` argument of `predict_fn` input.
      cache_final: Wether preprocesse dataset should be cached.
      pre_filter_fn: Predicate applied to the dataset for filtering records.
      first_class_name_only: Whether only the first class name should be
        considered (i.e. not using any aliases).
      dataset_overrides: Mapping `dataset_name` to an optional dictionary that
        can override parameters `dataset_name`, `data_dir`, `pp_img`, `pp_txt`,
        `class_names`, `split`, `pre_filter_fn`, and the extra
        `class_names_dataset_name`.
        Works with tuple/dict of tuples/dicts.
      async_delay: How many steps to wait before checking if all hosts have
        finished their batch. A value > 1 allows for more parallelized
        processing, but will results in more unnecessary steps with padded data.
    """
    t0 = time.monotonic()
    self.datasets = {}
    self.prompt_templates = prompt_engineering.get_prompt_templates(
        prompt_templates, canonicalize=canonicalize)
    self._axis_name = "batch"
    dataset_overrides = {k: dict(v) for k, v in dict(dataset_overrides).items()}

    for dataset_name in dataset_names:
      overrides = dataset_overrides.pop(dataset_name, {})
      dataset_name_ = overrides.pop("dataset_name", dataset_name)
      data_dir_ = overrides.pop("data_dir", data_dir)
      class_names_dataset_name = overrides.pop("class_names_dataset_name",
                                               dataset_name_)
      class_names_ = overrides.pop("class_names", class_names)
      class_names_ = prompt_engineering.get_class_names(
          dataset_name=class_names_dataset_name,
          source=class_names_,
          canonicalize=canonicalize)
      pp_img_ = overrides.pop("pp_img", pp_img)
      pp_txt_ = overrides.pop("pp_txt", pp_txt)
      cache_final_ = overrides.pop("cache_final", cache_final)
      split_ = overrides.pop("split", split)
      pre_filter_fn_ = overrides.pop("pre_filter_fn", pre_filter_fn)
      prompt_templates_ = overrides.pop("prompt_templates", prompt_templates)
      canonicalize_ = overrides.pop("canonicalize", canonicalize)
      prompt_templates_ = prompt_engineering.get_prompt_templates(
          prompt_templates_, canonicalize=canonicalize_)
      assert not overrides, f"Unknown overrides {dataset_name}: {overrides}"

      if first_class_name_only:
        class_names_ = [name.split(",")[0] for name in class_names_]
      ds_images, ds_texts = _split_and_batch(
          dataset_name=dataset_name_,
          data_dir=data_dir_,
          class_names=class_names_,
          batch_size=batch_size,
          split=split_,
          get_ds=functools.partial(
              prepare_datasets,
              pp_img=pp_img_,
              pp_txt=pp_txt_,
              cache_final=cache_final_,
              pre_filter_fn=pre_filter_fn_,
              prompt_templates=prompt_templates_))
      self.datasets[dataset_name] = dict(
          images=ds_images, texts=ds_texts, class_names=class_names_,
          dataset_name=dataset_name_, split=split_)

    assert not dataset_overrides, f"Extra overrides: {dataset_overrides}"

    def embed_texts(train_state, texts):
      """Returns text embeddings."""
      _, ztxt, _ = predict_fn(train_state, {"labels": texts})
      return ztxt

    def count_correct(train_state, return_embeddings, *, mask, labels, image,
                      ztxt):
      """Returns count of correct predictions (and optionally embeddings)."""
      zimg, _, _ = predict_fn(train_state, {"image": image})
      best_txt = (zimg @ ztxt.T).argmax(axis=1)
      # labels has format [[1, -1, -1], [5, -1, -1], [7, 2, -1], ...]
      # so here we count "any" correct, such that the counting matches the
      # multilabel scenario described in "are we done with imagenet"
      # (http://arxiv.org/abs/2006.07159) section 3.1
      if labels.ndim == 1:
        labels = labels[..., None]
      assert labels.ndim == 2, labels.shape
      matching = (best_txt[:, None] == labels).sum(axis=1)
      correct = jnp.where(mask, (matching > 0).astype(jnp.int32), 0).sum()
      correct = jnp.sum(correct)
      if return_embeddings:
        return correct, zimg
      else:
        return correct, None

    self.devices = devices
    self.mesh = jax.sharding.Mesh(devices, ("devices",))

    self._embed_texts_p = jax.jit(
        embed_texts, out_shardings=NamedSharding(self.mesh, P()))
    self._count_correct_p = jax.jit(count_correct, static_argnums=(1,),
                                    out_shardings=NamedSharding(self.mesh, P()))
    self._count_p = jax.jit(jnp.sum,
                            out_shardings=NamedSharding(self.mesh, P()))
    self._all_gather_p = jax.jit(
        lambda x: x, out_shardings=NamedSharding(self.mesh, P()))

    self._compiled = set()
    assert async_delay > 0, f"async_delay must be >0, not {async_delay}"
    self._async_delay = async_delay
    logging.info("Initialized evaluator in %.1f seconds", time.monotonic() - t0)

  def _embed_texts(self, train_state, dataset_name):
    """Returns per-class averaged text embeddings."""
    t0 = time.monotonic()
    logging.info("Starting text embedding...")
    ns = []
    embeddings = []
    data = {"label": [], "mask": []}

    ds_b = input_pipeline.start_global(
        self.datasets[dataset_name]["texts"], self.devices)
    for batch in ds_b:
      ns.append(jax.device_get(self._count_p(batch["mask"])))
      if len(ns) >= self._async_delay and ns[-self._async_delay] == 0:
        break

      embeddings.append(jax.device_get(self._embed_texts_p(
          train_state, batch["labels"])))
      for name in data:
        data[name].append(jax.device_get(self._all_gather_p(batch[name])))

      if self._embed_texts_p not in self._compiled:
        logging.info("Compiled text embeddings in %.1fs", time.monotonic() - t0)
        t0 = time.monotonic()
        self._compiled.add(self._embed_texts_p)

    ns = np.array(ns)
    n = ns.sum()
    data["embedding"] = embeddings
    data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    mask = data.pop("mask").astype(bool)
    data = {k: v[mask] for k, v in data.items()}
    data["average_embedding"] = _average_embeddings(
        data["embedding"],
        labels=data["label"],
        num_classes=len(self.datasets[dataset_name]["class_names"]),
        normalize=True)

    logging.info("Embedded %s text in %d steps - ...%s", dataset_name, len(ns),
                 ns[-10:])
    logging.info("Totalling %d text in %.1fs", n, time.monotonic() - t0)
    logging.info("Total texts embeddings size %.1fM",
                 data["embedding"].nbytes / 1e6)
    return data

  def evaluate(self,
               train_state,
               dataset_name,
               *,
               return_embeddings=False):
    """Returns evaluation results."""
    texts = self._embed_texts(train_state, dataset_name)
    ztxt_p = texts["average_embedding"]
    ztxt_p = utils.reshard(ztxt_p, NamedSharding(self.mesh, P()))

    t0 = time.monotonic()
    logging.info("Starting image embedding...")

    ns = []
    embeddings = []
    corrects = []
    data = {"mask": [], "label": []} if return_embeddings else {}

    ds_b = input_pipeline.start_global(
        self.datasets[dataset_name]["images"], self.devices)
    for batch in ds_b:
      ns.append(jax.device_get(self._count_p(batch["mask"])))
      if len(ns) >= self._async_delay and ns[-self._async_delay] == 0:
        break

      labels = batch["label"]
      correct_p, embs_p = self._count_correct_p(
          train_state,
          return_embeddings,
          mask=batch["mask"],
          labels=labels,
          image=batch["image"],
          ztxt=ztxt_p,
      )
      corrects.append(jax.device_get(correct_p))
      if self._count_correct_p not in self._compiled:
        logging.info("Compiled image embeddings in %.1fs",
                     time.monotonic() - t0)
        t0 = time.monotonic()
        self._compiled.add(self._count_correct_p)

      if return_embeddings:
        embeddings.append(jax.device_get(self._all_gather_p(embs_p)))
      for name in data:
        data[name].append(jax.device_get(self._all_gather_p(batch[name])))

    ns = np.array(ns)
    n = ns.sum()
    correct = np.array(corrects).sum()

    logging.info("Embedded %s image in %d steps - ...%s", dataset_name, len(ns),
                 ns[-10:])
    logging.info("Totalling %d image in %.1fs", n, time.monotonic() - t0)
    ret = {
        "accuracy": correct / n,
        "correct": correct,
        "count": n,
    }
    logging.info("Dataset %s, results %s", dataset_name, ret)

    if return_embeddings:
      data["embedding"] = embeddings
      data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
      logging.info("Total images embeddings size %.1fM",
                   data["embedding"].nbytes / 1e6)
      mask = data.pop("mask").astype(bool)
      ret["images"] = {k: v[mask] for k, v in data.items()}
      ret["texts"] = texts

    return ret

  def run(self, train_state):
    """Returns metrics."""
    return [(f"{dataset_name}_accuracy",
             self.evaluate(train_state, dataset_name)["accuracy"])
            for dataset_name in self.datasets]
