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

"""Evaluator to save predictions."""
# pylint: disable=consider-using-from-import
import functools
import io  # pylint: disable=unused-import
import itertools
import os

from absl import flags
from absl import logging
from big_vision import input_pipeline
from big_vision.datasets import core as ds_core
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import jax
import numpy as np

from tensorflow.io import gfile  # pylint: disable=unused-import

# Temporary global flag to facilitate backwards compatability.
API = 'jit'


# Note: global to avoid jax re-compiling across different evaluator instances.
@functools.cache
def _get_predict_fn(predict_fn, mesh=None):
  """Wrapper for jit-compiled predict function."""

  # `out_shardings` annotation is needed because of the `all_gather` ops in the
  # pmap implementation.
  @functools.partial(jax.jit,
                     out_shardings=jax.sharding.NamedSharding(
                         mesh, jax.sharding.PartitionSpec()))
  def _run_predict_fn(train_state, batch):
    """Run predict_fn and gather all outputs on all devices."""
    y = predict_fn(train_state, batch)
    return {'inputs': batch, 'outputs': y, 'mask': batch['_mask']}
  return _run_predict_fn


class Evaluator:
  """Save predictions in "{FLAGS.workdir}/{outfile}".

  Results can then be easily inspected in a notebook such as:

  ```
    results = utils.load_checkpoint("<full_path_to_outfile>")
    inputs, outputs = (results["inputs"], results["outputs"])
  ```
  """

  def __init__(self, predict_fn, pp_fn, batch_size, data, outfile,
               cache_final=True, cache_raw=False, prefetch=1, *, devices):
    self.predict_fn = _get_predict_fn(
        predict_fn, jax.sharding.Mesh(devices, ('devices',)))

    # Prepare data for each process and pad with zeros so all processes have the
    # same number of batches.
    data = ds_core.get(**data)
    self.dataset, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), batch_size=batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        preprocess_fn=pp_builder.get_preprocess_fn(pp_fn),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(
        self.dataset, devices, prefetch)

    self.path = os.path.join(flags.FLAGS.workdir, outfile)

  def run(self, train_state):
    """Compute all predictions, gather in main host and save in outfile."""
    count = 0
    outputs = []
    for batch in itertools.islice(self.data_iter, self.steps):
      out = self.predict_fn(train_state, batch)
      if jax.process_index():
        continue

      out = jax.device_get(out)
      # Note that we need to access `out['mask']` here `x` does not have that
      # field during the tree map.
      out = jax.tree_map(lambda x: x[out['mask']], out)  # pylint: disable=cell-var-from-loop
      count += out['mask'].shape[0]
      out.pop('mask')
      outputs.append(out)

      logging.log_every_n_seconds(
          logging.INFO, 'Save predictions: processed %i examples so far.', 30,
          count)

    if jax.process_index():
      return

    logging.info('Save predictions: processed %d examples.', count)

    # Actually save in filesystem.
    outputs = jax.tree_map(lambda *x: np.concatenate(x, axis=0), *outputs)
    names_and_vals, _ = u.tree_flatten_with_names(outputs)
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, **{k: v for k, v in names_and_vals})
    with gfile.GFile(self.path, 'wb') as f:
      f.write(io_buffer.getvalue())
    return

    yield None  # pylint: disable=unreachable
