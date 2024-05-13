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

"""Core data functions, dispatch calls to the requested dataset."""
import importlib


# Note: intentionally not using ABC to avoid forcing implementation of every
# method, since one can imagine train-only datasets for example.
class DataSource:
  """The API that any data source should implement."""

  def get_tfdata(self, ordered, *, process_split=True, allow_cache=True):
    """Creates this data object as a tf.data.Dataset.

    This will be called separately in each process, and it is up to the dataset
    implementation to shard it accordingly if desired!

    Args:
      ordered: if True, the dataset should use deterministic ordering, if False
        it may have undefined ordering. Think of True == val, False == train.
      process_split: if False then every process receives the entire dataset
        (e.g.  for evaluators running in a single process).
      allow_cache: whether to allow caching the opened data or not.

    Returns:
      A tf.data.Dataset object.

    Raises:
      RuntimeError: if not implemented by the dataset, but called.
    """
    raise RuntimeError("not implemented for {self.__class__.__name__}")

  @property
  def total_examples(self):
    """Returns number of examples in the dataset, regardless of sharding."""
    raise RuntimeError("not implemented for {self.__class__.__name__}")

  def num_examples_per_process(self):
    """Returns a list of the numer of examples for each process.

    This is only needed for datasets that should go through make_for_inference.

    Returns:
      Returns a list of the numer of examples for each process.

      Ideally, this would always be `[total() / nprocess] * nprocess`, but in
      reality we can almost never perfectly shard a dataset across arbitrary
      number of processes.

      One alternative option that can work in some cases is to not even shard
      the dataset and thus return `[num_examples()] * nprocess.

    Raises:
      RuntimeError: if not implemented by the dataset, but called.
    """
    raise RuntimeError("not implemented for {self.__class__.__name__}")


def get(name, **kw):
  if name.startswith("bv:"):
    mod = importlib.import_module(f"big_vision.datasets.{name[3:]}")
    return mod.DataSource(**kw)
  else:
    mod = importlib.import_module("big_vision.datasets.tfds")
    return mod.DataSource(name, **kw)
