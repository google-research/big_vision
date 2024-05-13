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

"""Load and run the PaliGemma model."""
import functools
import sys

from absl import app
from absl import flags
from absl import logging

# pylint: disable=all
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import ml_collections
import numpy as np

import big_vision.models.proj.paligemma.gemma_bv
import big_vision.models.proj.paligemma.paligemma as model_mod
import big_vision.models.vit
import big_vision.pp.builder
import big_vision.pp.tokenizer
import big_vision.pp.ops_image
import big_vision.pp.ops_general
import big_vision.pp.ops_text
import big_vision.pp.proj.paligemma.ops
import big_vision.sharding
import big_vision.trainers.proj.paligemma.predict_fns
import big_vision.utils as u
# pylint: enable=all

# We always want to be explicit about any host-device transfers.
jax.config.update("jax_transfer_guard", "disallow")

CKPT = flags.DEFINE_string(
    "ckpt", default=None, help="Path to checkpoint.")
IMAGE = flags.DEFINE_string(
    "image", default=None, help="Path to input image.")

SAMPLER = flags.DEFINE_string(
    "sampler", default="greedy", help="Decoding strategy. Try `nucleus(0.1)`")
RES = flags.DEFINE_integer(
    "res", default=224, help="Image resolution (224, 448, 896).")
MAX_DECODE_LEN = flags.DEFINE_integer(
    "max_decode_len", default=128, help="Max total generation steps.")
PREFILL_LEN = flags.DEFINE_integer(
    "prefill_len", default=32, help="Size of prefill (prompt). "
    "Shorter is faster, but too short will cut off your prompt.")

TOKENIZER = "gemma(tokensets=['loc', 'seg'])"


def load_model(ckpt):
  model_cfg = ml_collections.FrozenConfigDict(dict(
      img=dict(variant="So400m/14", pool_type="none", scan=True),
      llm=dict(vocab_size=256_000 + 1024 + 128),
  ))
  model = model_mod.Model(**model_cfg)
  params = model_mod.load(None, ckpt, model_cfg)
  return model, params


def info(s, *a):
  logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  logging.flush()


def main(argv):
  info(f"{argv=}")
  info("Loading model...")
  model, params = load_model(CKPT.value)

  predict_fns = big_vision.trainers.proj.paligemma.predict_fns.get_all(model)

  info("Loading tokenizer...")
  tokzr = big_vision.pp.tokenizer.get_tokenizer(TOKENIZER)

  info("Creating mesh and sharding params...")
  mesh = Mesh(jax.devices(), ("data"))
  repl_sharding = NamedSharding(mesh, PartitionSpec())
  data_sharding = NamedSharding(mesh, PartitionSpec("data"))
  params_sharding = big_vision.sharding.infer_sharding(
      params, strategy=[(".*", "fsdp(axis='data')")], mesh=mesh)

  # Ship the params to device(s)
  params = jax.tree.map(lambda x, sh: u.reshard(x, sh), params, params_sharding)

  # Mostly go through pp ops to build our batch:
  pp_fn = big_vision.pp.builder.get_preprocess_fn("|".join([
      f"decode|resize({RES.value})|value_range(-1, 1)",
      f"tok(key='prefix', bos='yes', model={repr(TOKENIZER)})",
      f"tok(key='septok', text='\\n', model={repr(TOKENIZER)})",
      'masked_concat(["prefix", "septok"], mask_ar=[0, 0], mask_input=[1, 1])',
      f'tolen({PREFILL_LEN.value}, pad_value=0, key="text")',
      f'tolen({PREFILL_LEN.value}, pad_value=1, key="mask_ar")',
      f'tolen({PREFILL_LEN.value}, pad_value=0, key="mask_input")',
      'keep("image", "text", "mask_ar", "mask_input")',
  ]), log_data=False)

  decode = functools.partial(
      predict_fns["decode"], devices=jax.devices(),
      eos_token=tokzr.eos_token, max_decode_len=MAX_DECODE_LEN.value,
      sampler=SAMPLER.value)

  def make_batch(fname, prompt):
    image = open(fname, "rb").read()

    # Create an example
    example = pp_fn({"image": image, "prefix": np.array(prompt)})
    example["_mask"] = np.array(True)  # True means valid non-pad example

    batch = jax.tree.map(lambda x: x[None], example)
    return u.reshard(batch, repl_sharding)  # Move to device(s)

  info("Precompiling inference function...")
  decode({"params": params}, batch=make_batch(IMAGE.value, "caption en"))

  info("Type a prompt and press enter, for example 'caption en': ")
  for line in map(str.strip, sys.stdin):
    tokens = decode({"params": params}, batch=make_batch(IMAGE.value, line))
    tokens = jax.device_get(tokens)[0]  # First batch entry.

    # TODO: b/lbeyer - flip around: output on stdout, logs on stderr.
    print(tokzr.to_str(tokens), file=sys.stderr, flush=True)


if __name__ == "__main__":
  flags.mark_flag_as_required("ckpt")
  flags.mark_flag_as_required("image")
  app.run(main)
