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

"""Evaluator to save predictions."""
# pylint: disable=consider-using-from-import
import os

from absl import flags
from absl import logging
import big_vision.evaluators.proj.uvim.common as common
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import jax
import numpy as np
import tensorflow as tf


class Evaluator:
  """Save predictions in "{FLAGS.workdir}/{outfile}".

  Results can then be easily inspected in a notebook such as:

  ```
    results = utils.load_checkpoint(None, "<full_path_to_outfile>")
    inputs, outputs = (results["inputs"], results["outputs"])
  ```
  """

  def __init__(self, predict_fn, pp_fn, dataset, split, batch_size, outfile,
               predict_kwargs=None, dataset_dir=None):
    # Prepare to run predict on all processes and gather predictions on all
    # devices. Note: if needed consider only gather across processes.
    def predict(params, batch):
      y = predict_fn(params, batch['inputs'], **(predict_kwargs or {}))
      res = {'inputs': batch['inputs'], 'outputs': y, 'mask': batch['mask']}
      return jax.lax.all_gather(res, axis_name='data', axis=0, tiled=True)

    self.predict_fn = jax.pmap(predict, axis_name='data')

    # Prepare data for each process and pad with zeros so all processes have the
    # same number of batches.
    def preprocess(example):
      return {
          'mask': tf.constant(1),
          'inputs': pp_builder.get_preprocess_fn(pp_fn)(example),
      }
    self.data = common.get_jax_process_dataset(
        dataset=dataset, split=split,
        dataset_dir=dataset_dir,
        global_batch_size=batch_size,
        pp_fn=preprocess)

    self.path = os.path.join(flags.FLAGS.workdir, outfile)

  def run(self, params):
    """Compute all predictions, gather in main host and save in outfile."""
    count = 0
    outputs = []
    for batch in self.data.as_numpy_iterator():
      out = self.predict_fn(params, batch)
      if jax.process_index():
        continue

      out = jax.device_get(jax.tree_map(lambda x: x[0], out))
      out = jax.tree_map(lambda x: x[out['mask'] == 1], out)  # pylint: disable=cell-var-from-loop
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
    u.save_checkpoint(outputs, self.path, compressed=True)
    return

    yield None  # pylint: disable=unreachable
