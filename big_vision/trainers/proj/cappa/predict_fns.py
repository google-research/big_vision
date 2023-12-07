# Copyright 2023 Big Vision Authors.
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

"""Prediction functions for clippo/generative.py."""

import functools

import big_vision.pp.ops_text as pp_ops_text
import big_vision.utils as u
import jax
import jax.numpy as jnp
import numpy as np

# pylint: disable=missing-function-docstring


# We do not jit/pmap this function, because it is passed to evaluator that
# does it later. We output as many intermediate tensors as possible for
# maximal flexibility. Later `jit` will prune out things that are not needed.
def predict_fn_perplexity(
    train_state, batch, *, model):
  logits = model.apply(
      {"params": train_state["params"]},
      batch["image"],
      batch["labels"],
      train=False,
  )
  return logits, {"logits": logits}


def predict_fn_enc_rep(train_state, batch, *, model):
  logits, out = model.apply(
      {"params": train_state["params"]},
      batch["image"],
      None,
      train=False,
      return_enc_features=True,
  )
  return logits, out


def predict_fn_score(
    train_state, batch, *, model, prompt="", prompt_tokenizer=""):
  """For a batch of images, return score (LL) for each image-label pair."""
  encoded = model.apply(
      {"params": train_state["params"]},
      batch["image"],
      train=False,
      method=model.encode,
  )

  # This needs to be added by the evaluator. It is the pre-computed tokenized
  # list of all available labels. For ImageNet-1k, that's (1000, 13).
  all_labels = batch["_label_tokens"]

  # Optionally prefix a single prompt to all labels:
  if prompt and prompt_tokenizer:
    prompt = make_prompt(prompt, prompt_tokenizer)  # Note: this is cached.
    prompts = jnp.tile(prompt, (all_labels.shape[0], 1))
    all_labels = jnp.concatenate([prompts, all_labels], axis=-1)
    # For ImageNet-1k and a prompt of length 2, we now have (1000, 15).

  def score_label(label):
    """Score (LogLik) each minibatch example (image) with a single `label`."""
    label_rep = jnp.tile(label, (encoded.shape[0], 1))
    logits = model.apply(
        {"params": train_state["params"]},
        encoded,
        label_rep,
        train=False,
        decode=False,
        method=model.decode,
    )
    # The returned value is (batch,) scalars, the score each image has with
    # this label. We turn the softmax_xent's NLL into LL so higher = better.
    return -u.weighted_softmax_xent(
        logits=logits,
        labels=label_rep,
        weights=(label_rep > 0).astype(jnp.float32),  # Ignore <PAD> (=0).
        reduction=False,
        normalize=False,
    )

  # Use lax.map() instead of vmap() to conserve memory.
  nlls = jax.lax.map(score_label, all_labels)  # -> (nlabel, batch)
  return nlls.T  # -> (batch, nlabel) array of scores.


@functools.cache
def make_prompt(prompt, tokenizer_path, seq_len=None):
  """Tokenizes `prompt` with specified tokenizer, with optional padding."""
  tokenizer = pp_ops_text.create_tokenizer(tokenizer_path, add_eos=False)

  prompt = tokenizer.tokenize(prompt).numpy()
  if seq_len:
    prompt = np.pad(prompt, (0, seq_len - len(prompt))).astype(np.int32)
  return prompt


def get_predict_fns(model):
  """Returns `predict_fns` for evaluators."""
  fns = {
      "perplexity": predict_fn_perplexity,
      "score": predict_fn_score,
      "enc_rep": predict_fn_enc_rep,
  }
  return {name: functools.partial(fn, model=model) for name, fn in fns.items()}
