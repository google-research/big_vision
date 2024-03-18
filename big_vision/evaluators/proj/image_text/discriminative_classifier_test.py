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

"""Tests for discriminative_classifier."""

from unittest import mock

from big_vision.evaluators.proj.image_text import discriminative_classifier
from big_vision.pp import ops_general  # pylint: disable=unused-import
from big_vision.pp import ops_image  # pylint: disable=unused-import
from big_vision.pp.registry import Registry
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@Registry.register("preprocess_ops.test_texts2labels")
def _get_test_texts2labels():

  def pp(features):
    features["labels"] = tf.strings.to_number(features["texts"])
    return features

  return pp


@Registry.register("preprocess_ops.copy_from")
def _get_copy_from(**key_map):

  def copy_from(d):
    d = dict(d)
    for k1, k2 in key_map.items():
      d[k1] = d[k2]
    return d

  return copy_from


class _Model(nn.Module):

  @nn.compact
  def __call__(self, image, texts):
    self.param("x", lambda _: 0.)

    def z(x):
      if x is not None:
        # Note that the returned vector is most similar with other vectors
        # generated from the same underlying `x[:]`.
        return jnp.stack([jnp.cos(x / 10.), jnp.sin(x / 10.)]).T

    if texts is not None:
      texts %= 5  # For testing `pre_filter_fn` below.
    return z(image), z(texts), None


  class DiscriminativeClassifierTest(tf.test.TestCase):

  def test_prepare_datasets(self):

    def generator():
      yield {
          "image": tf.ones([5, 5, 3], tf.float32),
          "label": 1,
      }
      yield {
          "image": tf.ones([4, 4, 3], tf.float32),
          "label": 2,
      }

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            "image": tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            "label": tf.TensorSpec(shape=[], dtype=tf.int64),
        })
    class_names = [
        "class1,class1a",
        "class2",
    ]
    prompt_templates = [
        "test {}",
        "test {} test",
    ]
    ds_img, ds_txt = discriminative_classifier.prepare_datasets(
        ds,
        class_names,
        prompt_templates=prompt_templates,
        pp_img="resize(2)",
        pp_txt="copy_from(labels='texts')",
    )

    it_img = iter(ds_img)
    batch = next(it_img)
    self.assertAllEqual(1, batch["label"])
    self.assertAllEqual(tf.ones([2, 2, 3]), batch["image"])
    batch = next(it_img)
    self.assertAllEqual(2, batch["label"])
    self.assertAllEqual(tf.ones([2, 2, 3]), batch["image"])

    it_txt = iter(ds_txt)
    batch = next(it_txt)
    self.assertAllEqual(0, batch["label"])
    self.assertAllEqual("test class1", batch["labels"])
    batch = next(it_txt)
    self.assertAllEqual(0, batch["label"])
    self.assertAllEqual("test class1 test", batch["labels"])
    batch = next(it_txt)
    self.assertAllEqual(0, batch["label"])
    self.assertAllEqual("test class1a", batch["labels"])
    batch = next(it_txt)
    self.assertAllEqual(0, batch["label"])
    self.assertAllEqual("test class1a test", batch["labels"])
    batch = next(it_txt)
    self.assertAllEqual(1, batch["label"])
    self.assertAllEqual("test class2", batch["labels"])
    batch = next(it_txt)
    self.assertAllEqual(1, batch["label"])
    self.assertAllEqual("test class2 test", batch["labels"])

  def test_average_embeddings(self):
    self.assertAllEqual(jnp.array([
        [2.], [4.], [8.],
    ]), discriminative_classifier._average_embeddings(
        embeddings=jnp.array([
            1., 3., 3., 1.,  # label1
            8., 0.,  # label2
            32., 0., 0., 0.,  # label3
        ])[..., None],
        labels=jnp.array([
            0, 0,  # label1
            0, 0,  # label1 (alias)
            1, 1,  # label2
            2, 2,  # label3
            2, 2,  # label3 (alias)
        ], jnp.int32),
        num_classes=3, normalize=False))
    self.assertAllEqual(
        jnp.array([
            [2**-.5, 2**-.5],
        ]),
        discriminative_classifier._average_embeddings(
            embeddings=jnp.array([[2., 2.]]),
            labels=jnp.array([0], jnp.int32),
            num_classes=1,
            normalize=True))

  @mock.patch("big_vision.evaluators.proj."
              "image_text.prompt_engineering.get_class_names")
  @mock.patch("big_vision.evaluators.proj."
              "image_text.prompt_engineering.get_prompt_templates")
  @mock.patch("big_vision.evaluators.proj."
              "image_text.discriminative_classifier._get_dataset_info")
  def test_evaluate(self, get_dataset_info_mock, get_prompt_templates_mock,
                    get_class_names_mock):
    per_device_batch_size = 10  # Make sure we have some unfiltered examples.
    global_batch_size = per_device_batch_size * jax.device_count()
    per_host_num_examples = int(
        np.ceil(global_batch_size / jax.process_count()))
    splits = {
        "test":
            tfds.core.SplitInfo(
                name="test", shard_lengths=[per_host_num_examples], num_bytes=0)
    }

    model = _Model()
    params = model.init(jax.random.PRNGKey(0), None, None)["params"]

    prompt_templates = [
        "test prompt 1 {}",
        "test prompt 2 {}",
    ]
    class_names = [
        f"test_class_{i}" for i in range(10)
    ]

    get_prompt_templates_mock.return_value = prompt_templates
    get_class_names_mock.return_value = class_names
    get_dataset_info_mock.return_value.splits = splits

    def pre_filter_fn(features):
      return features["label"] < 5  # matches `texts %= 5` above

    dataset_name = "cifar10_test"
    with tfds.testing.mock_data(num_examples=per_host_num_examples):
      evaluator = discriminative_classifier.Evaluator(
          lambda p, b: model.apply({"params": p},
                                   b.get("image", None),
                                   b.get("labels", None)),
          dataset_names=[dataset_name],
          prompt_templates="test_prompts",
          batch_size=global_batch_size,
          devices=jax.devices(),
          pp_img="copy_from(image='label')",
          pp_txt="copy_from(labels='label')",
          dataset_overrides={
              dataset_name: {
                  "dataset_name": "cifar10",
                  "class_names": "test_classes",
                  "pre_filter_fn": pre_filter_fn,
              }
          },
          first_class_name_only=True,
      )
      results = evaluator.evaluate(
          params,
          dataset_name,
          return_embeddings=True)
      metrics = dict(evaluator.run(params))

    # Assert all examples were processed.
    self.assertLen(results["texts"]["embedding"],
                   len(class_names) * len(prompt_templates))
    self.assertLen(results["texts"]["average_embedding"], len(class_names))
    self.assertAllEqual(
        sorted(results["texts"]["label"]),
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    # Note that above model makes perfect predictions by design.
    self.assertEqual(1.0, results["accuracy"])
    self.assertEqual(1.0, metrics[f"{dataset_name}_accuracy"])


if __name__ == "__main__":
  tf.test.main()
