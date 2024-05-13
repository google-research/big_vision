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

"""Evaluator that save inputs and outputs of prediction functions."""
import functools

from absl import flags
from absl import logging

from big_vision import input_pipeline
from big_vision import optax as bv_optax
from big_vision import utils
from big_vision.datasets import core as ds_core
from big_vision.pp import builder as pp_builder

import jax
import numpy as np

# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# Note: global to avoid jax re-compiling across different evaluator instances.
def _run_predict_fn(predict_fn, train_state, batch):
  """Run predict_fn and gather all outputs on all devices."""
  y = predict_fn(train_state, batch)
  return {'inputs': batch, 'outputs': y}


class Evaluator:
  """Evaluator that saves the inputs and outputs of a prediction function.

  Example configuration:

  ```
    config.evals.save_pred = {
      'type': 'save',
      'pred': 'inference',
      'outfile': '{workdir}/inference-{step:09d}.npz',
      'data': ..., 'pp_fn': ..., 'log_steps': ...,
  }
  ```

  Results can then be easily inspected in a notebook such as:

  ```
    results = utils.load_checkpoint("<full_path_to_outfile>")
    inputs, outputs = (results["inputs"], results["outputs"])
  ```
  """

  def __init__(self, predict_fn, data, pp_fn, batch_size, outfile,
               cache_final=True, cache_raw=False, prefetch=1, *, devices):
    replicate = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, ('devices',)),
        jax.sharding.PartitionSpec()
    )
    self.predict_fn = functools.partial(
        jax.jit(_run_predict_fn, static_argnums=0, out_shardings=replicate),
        predict_fn,
    )

    data = ds_core.get(**data)
    self.dataset, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True),
        batch_size=batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        preprocess_fn=pp_builder.get_preprocess_fn(pp_fn),
        cache_final=cache_final,
        cache_raw=cache_raw,
    )
    self.data_iter = input_pipeline.start_global(
        self.dataset, devices, prefetch
    )

    self.outfile = outfile

  def run(self, train_state):
    """Compute all predictions, gather in main host and save in outfile."""
    step = jax.device_get(bv_optax.get_count(train_state['opt'], jittable=True))
    outfile = self.outfile.format(workdir=flags.FLAGS.workdir, step=step)

    count = 0
    outputs = []
    for _, batch in zip(range(self.steps), self.data_iter):
      out = self.predict_fn(train_state, batch)
      if jax.process_index():
        continue

      out = jax.device_get(out)
      mask = out['inputs']['_mask']
      out = jax.tree.map(lambda x: x[mask == 1], out)  # pylint: disable=cell-var-from-loop
      count += mask.shape[0]
      out['inputs'].pop('_mask')
      outputs.append(out)

      logging.log_every_n_seconds(
          logging.INFO, 'Processed %i examples so far.', 60,
          count)

    if jax.process_index():
      return

    logging.info('Saving %d examples in %s', count, outfile)
    outputs = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *outputs)
    utils.save_checkpoint(outputs, outfile, compressed=True)
    return

    yield None  # pylint: disable=unreachable
