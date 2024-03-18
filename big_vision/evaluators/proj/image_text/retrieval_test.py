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

"""Tests for retrieval."""

from unittest import mock

from big_vision.evaluators.proj.image_text import retrieval
from big_vision.pp import ops_general  # pylint: disable=unused-import
from big_vision.pp import ops_image  # pylint: disable=unused-import
from big_vision.pp import registry
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def _get_test_texts2labels():

  def pp(features):
    features["labels"] = tf.strings.to_number(features["texts"])
    return features

  return pp


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
        batch_size = len(x)
        # Note that the returned vector is most similar with other vectors
        # generated from the same underlying `x[:]`.
        x = jnp.concatenate([100 * jnp.ones([batch_size, 1]), x[:, None]],
                            axis=1)
        return x / jnp.linalg.norm(x, axis=1)[:, None]

    return z(image), z(texts), None


def setUpModule():
  chex.set_n_cpu_devices(8)


class RetrievalTest(tf.test.TestCase):

  def test_prepare_datasets(self):

    def generator():
      yield {
          "image": tf.ones([5, 5, 3], tf.float32),
          "captions": {
              "text": tf.constant(["11", "12"])
          }
      }
      yield {
          "image": tf.ones([4, 4, 3], tf.float32),
          "captions": {
              "text": tf.constant(["21", "22", "23"])
          }
      }

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            "image": tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            "captions": {
                "text": tf.TensorSpec(shape=[None], dtype=tf.string),
            },
        })
    with registry.temporary_ops(test_texts2labels=_get_test_texts2labels):
      ds_img, ds_txt = retrieval.prepare_datasets(
          ds,
          pp_img="resize(2)",
          pp_txt="test_texts2labels()",
          txt_name=("captions", "text"),
      )
    it_img = iter(ds_img)
    it_txt = iter(ds_txt)
    batch = next(it_img)
    self.assertAllEqual(batch["id"], 0)
    self.assertAllEqual(batch["image"], tf.ones([2, 2, 3]))
    batch = next(it_img)
    self.assertAllEqual(batch["id"], 1)
    self.assertAllEqual(batch["image"], tf.ones([2, 2, 3]))
    batch = next(it_txt)
    self.assertAllEqual(batch["id"], 0)
    self.assertAllEqual(batch["caption_i"], 0)
    self.assertAllEqual(batch["labels"], 11.0)
    batch = next(it_txt)
    self.assertAllEqual(batch["id"], 0)
    self.assertAllEqual(batch["caption_i"], 1)
    self.assertAllEqual(batch["labels"], 12.0)
    batch = next(it_txt)
    self.assertAllEqual(batch["id"], 1)
    self.assertAllEqual(batch["caption_i"], 0)
    self.assertAllEqual(batch["labels"], 21.0)
    batch = next(it_txt)
    self.assertAllEqual(batch["id"], 1)
    self.assertAllEqual(batch["caption_i"], 1)
    self.assertAllEqual(batch["labels"], 22.0)
    batch = next(it_txt)
    self.assertAllEqual(batch["id"], 1)
    self.assertAllEqual(batch["caption_i"], 2)
    self.assertAllEqual(batch["labels"], 23.0)

  def test_evaluate(self):
    per_device_batch_size = 2
    batch_size = per_device_batch_size * jax.device_count()
    num_examples = 1 * batch_size + 1
    splits = {
        "test":
            tfds.core.SplitInfo(
                name="test", shard_lengths=[num_examples], num_bytes=0)
    }

    model = _Model()
    params = model.init(jax.random.PRNGKey(0), None, None)["params"]

    with tfds.testing.mock_data(num_examples=num_examples):
      info_mock = mock.Mock()
      info_mock.splits = splits
      with mock.patch.object(retrieval, "_get_dataset_info",
                             lambda _: info_mock):
        with registry.temporary_ops(copy_from=_get_copy_from):
          evaluator = retrieval.Evaluator(
              lambda p, b: model.apply({"params": p},
                                       b.get("image", None),
                                       b.get("labels", None)),
              dataset="coco_captions",
              batch_size=batch_size,
              devices=jax.devices(),
              txt_name=("captions", "text"),
              pp_img="copy_from(image='id')",
              pp_txt="copy_from(labels='id')",
          )
      results = evaluator.evaluate(params)

    # Assert all examples were processed.
    self.assertLen(results["images"]["embeddings"], num_examples)
    self.assertLen(results["images"]["id"], num_examples)
    # Assert no padding was processed (expects exactly one (=first) image.id=0
    self.assertEqual((results["images"]["id"] == 0).sum(), 1)
    # Expect perfect ITR with above _Model()...
    self.assertEqual(results["img2txt"]["Recall@1"], 1.0)
    self.assertEqual(results["txt2img"]["Recall@5"], 1.0)


if __name__ == "__main__":
  tf.test.main()
