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

"""Decode autoregressive/bidirectional masked transformers.


Currently, we implement MaskGIT style temperature sampling:

In each step:
1. Get P = model(inputs), predicted GMMs
2. Get samples = sample_from(P)
3. Get probs = P[samples], ie, model evaluated at samples.
   We use this now as a confidence metric, but we scale the probs:
4. probs = probs ^ 1/choice_temperature
4. set probs[already_uncovered_points] = inf, ie, we will always keep
   uncovered points (no resampling!)
5. Now pick top K points from probs to keep for the next steps, where
   K = some monotonically increasing ratio of points as we go along decoding
"""

import dataclasses
from typing import Literal

from absl import logging
from big_vision.models.proj.givt import givt
import distrax
import flax
import jax
import jax.numpy as jnp


_CONFIDENCE_OF_KNOWN_TOKENS = jnp.inf


@jax.vmap
def _get_per_batch_mask(arr, k):
  (d,) = arr.shape
  indices = jnp.argsort(arr)
  valid_indices = jnp.arange(d) < k
  return jnp.zeros((d,), jnp.bool_).at[indices].set(valid_indices)


def _get_bottom_k_mask(arr, k):
  *leading, d = arr.shape
  arr = arr.reshape((-1, d))
  mask = _get_per_batch_mask(arr, k)
  return mask.reshape(*leading, -1)


def mask_by_random_topk(rng, mask_len, probs, temperature=1.0):
  """Create a mask.

  Adaption of jax.random.choice where probabilities are changed by scaling with
  `temperature` (probs = probs ^ (1/temperature)).

  Additionally, this function returns a mask of tokens to mask out, which
  are picked to be the low confidence ones. Thus, this function is roughly
  equivalent to (but not exactly at edge cases such as prob = inf..):

    keep = jax.random.choice(
        rng, seq_len,
        shape=(seq_len - mask_len,),
        # NOTE: probabilities are updated with `temperature`.
        p=jnp.power(probs, 1/temperature),
        replace=False
    )
    mask = jnp.ones((seq_len,), dtype=jnp.bool_)
    return mask.at[..., keep].set(False)

  Args:
    rng: a PRNG key used as the random key.
    mask_len: the number to mask.
    probs: the probabilities associated with each entry.
    temperature: when temperature = 1.0, it's identical to jax's implementation.
      The larger this value is, the more random the masking is picked.

  Returns:
    A binary masking map [batch_size, seq_len]. Contains True where we should
    mask (at mask_len locations), and False where we should keep.
  """
  confidence = jnp.log(probs) + temperature * jax.random.gumbel(
      rng, probs.shape)
  return _get_bottom_k_mask(confidence, mask_len)


@flax.struct.dataclass
class DecodeState:
  """Holds decoding state data."""

  rng: jax.Array  # Sampling random state.
  # The position of the decoding loop in the length dimension. Scalar int32.
  step: jax.Array
  # What we input at each step. Starts from all masks and is uncovered by
  # sampling. Note that this is an array with leading
  # dimension `num_steps + 1` because we start with all masked tokens and then
  # need `num_steps` to uncover all, i.e., the final output is given by
  # all_inputs_q[-1, ...].
  all_inputs_q: jax.Array  # float32 [num_steps + 1, batch, seq_len, c]
  # Has a 1 for every _uncovered_ point.
  uncovered_per_step: jax.Array  # bool_ [num_steps, batch, seq_len]
  logits_per_step: jax.Array  # [num_steps, batch, seq_len, num_logits]
  uncond_logits_per_step: jax.Array  # [num_steps, batch, seq_len, num_logits]
  prob_per_step: jax.Array  # Probability per step.
  # If CFG: Rejection sampling success rate.
  rejection_sampling_success_per_step: jax.Array

  @classmethod
  def make(
      cls,
      initial_rng: jax.Array,
      all_masked_input: jax.Array,
      num_logits: int,
      num_steps: int,
  ) -> "DecodeState":
    """Creates the initial state."""
    b, seq_len, c = all_masked_input.shape
    all_inputs_q = jnp.broadcast_to(
        all_masked_input,
        (num_steps + 1, b, seq_len, c),
    )
    return cls(
        initial_rng,
        step=jnp.array(0),
        all_inputs_q=all_inputs_q,
        uncovered_per_step=jnp.full((num_steps, b, seq_len), False, jnp.bool_),
        logits_per_step=jnp.full(
            (num_steps, b, seq_len, num_logits), jnp.nan, jnp.float32
        ),
        uncond_logits_per_step=jnp.full(
            (num_steps, b, seq_len, num_logits), jnp.nan, jnp.float32
        ),
        prob_per_step=jnp.full((num_steps, b, seq_len), jnp.nan, jnp.float32),
        rejection_sampling_success_per_step=jnp.full(
            (num_steps,), jnp.nan, jnp.float32
        ),
    )

  @property
  def current_inputs_q(self) -> jax.Array:
    """Returns the current quantized input."""
    return self.all_inputs_q[self.step, ...]

  @property
  def num_steps(self) -> int:
    """Returns number of decode steps."""
    return self.uncovered_per_step.shape[0]

  def _steps_mask(self) -> jax.Array:
    return jnp.arange(self.num_steps) <= self.step

  @property
  def total_uncovered(self) -> jax.Array:
    """Returns the total uncovered mask up to and including current step."""
    return self.uncovered_per_step.sum(
        axis=0, where=self._steps_mask()[:, jnp.newaxis, jnp.newaxis]
    ).astype(jnp.bool_)

  def split_rng(self) -> tuple["DecodeState", jax.Array]:
    """Splits of RNG for the current step."""
    rng, step_rng = jax.random.split(self.rng, 2)
    return self.replace(rng=rng), step_rng

  def set_next_input(self, next_input_q: jax.Array) -> "DecodeState":
    """Sets the input for the next step."""
    return self._set_row("all_inputs_q", self.step + 1, next_input_q)

  def set_uncover_at_current_step(self, uncovered: jax.Array) -> "DecodeState":
    """Sets what was uncovered after the current step."""
    return self._set_row("uncovered_per_step", self.step, uncovered)

  def set_logits_at_current_step(self, logits: jax.Array) -> "DecodeState":
    return self._set_row("logits_per_step", self.step, logits)

  def set_uncond_logits_at_current_step(
      self, logits: jax.Array
  ) -> "DecodeState":
    return self._set_row("uncond_logits_per_step", self.step, logits)

  def set_rejection_sampling_success_at_current_step(
      self, success: jax.Array
  ) -> "DecodeState":
    return self._set_row(
        "rejection_sampling_success_per_step", self.step, success
    )

  def set_prob_at_current_step(self, prob: jax.Array) -> "DecodeState":
    return self._set_row("prob_per_step", self.step, prob)

  def increment_step(self) -> "DecodeState":
    """Increments step."""
    return self.replace(step=self.step + 1)

  def _set_row(self, attr_name, row_index, row_value):
    """Sets one row of the variables that have shape (num_steps, ...)."""
    current_value = getattr(self, attr_name)
    _, *expected_shape = current_value.shape
    if row_value.shape != tuple(expected_shape):
      raise ValueError(f"Expected {row_value.shape} == {expected_shape}!")
    if row_value.dtype != current_value.dtype:
      raise ValueError(f"Expected {row_value.dtype} == {current_value.dtype}")
    new_value = current_value.at[row_index, ...].set(row_value)
    return self.replace(**{attr_name: new_value})


@dataclasses.dataclass(frozen=True)
class MaskedGenerationConfig:
  """Config for masked generation.

  Attributes:
    num_steps: Number of sampling steps.
    should_anneal_temperature: If given, anneal choice temperature as we go
      through the sampling steps.
    choice_temperature: Temperature for picking points.
    ordering: How to order to select. Supports:
      maskgit: Maskgit style, use P[samples]
    schedule: Inference mask schedule.
    cfg_inference_weight: CFG Inference weight.
  """
  num_steps: int = 16
  should_anneal_temperature: bool = True
  choice_temperature: float = 1.0
  ordering: Literal["maskgit"] = "maskgit"
  schedule: str = "cosine"
  cfg_inference_weight: float = 0.0


def _assert_single_component_get_loc_scale(
    pdf: distrax.Distribution, rng=None, mixture=None
):
  """Extracts loc and scale from a single mixture GMM."""
  if not isinstance(pdf, distrax.MixtureSameFamily):
    raise ValueError(f"Expected mixture! Got {type(pdf)}")
  components_d = pdf.components_distribution
  if isinstance(components_d, distrax.MultivariateNormalDiag):
    loc, scale_diag = components_d.loc, components_d.scale_diag
    b, s, m, _ = loc.shape
    if mixture is None:
      assert rng is not None
      # Shape (b, seq)
      mixture = pdf.mixture_distribution.sample(seed=rng)
      mixture = jax.nn.one_hot(mixture, num_classes=m, axis=-1)
    assert mixture.shape == (b, s, m), (mixture.shape, loc.shape)
    loc = (loc * mixture[..., None]).sum(-2)
    scale_diag = (scale_diag * mixture[..., None]).sum(-2)
    return loc, scale_diag, mixture
  else:
    loc, scale = components_d.loc, components_d.scale
    if loc.shape[-1] != 1 or scale.shape[-1] != 1:
      raise ValueError(f"Expected one mixture! {loc.shape}/{scale.shape}")
    return loc[..., 0], scale[..., 0], None


class CFGDensity:
  """Helper to get probability and samples via CFG."""

  pdf_c: distrax.Distribution
  pdf_u: distrax.Distribution
  w: float
  simple: distrax.Distribution
  fac: jax.Array

  def __init__(
      self,
      pdf_c: distrax.Distribution,
      pdf_u: distrax.Distribution,
      w: float,
      rng: jax.Array,
  ) -> None:
    loc_c, scale_c, mixture = _assert_single_component_get_loc_scale(pdf_c, rng)
    # Note: RNG only needed when we have mixtures, to select components.
    loc_u, scale_u, _ = _assert_single_component_get_loc_scale(
        pdf_u, rng, mixture=mixture
    )

    # Definitly wider than whatever we had before. The mean should be slightly
    # away though!
    loc_simple = loc_c
    scale_simple = jnp.stack([scale_c, scale_u], -1).max(-1) * 2
    self.simple = distrax.Normal(loc_simple, scale_simple)

    self.pdf_c = distrax.Normal(loc_c, scale_c)
    self.pdf_u = distrax.Normal(loc_u, scale_u)
    self.w = w

    assert loc_c.ndim == 3, loc_c.shape
    points = loc_c[jnp.newaxis, ...] + jnp.linspace(-10, 10, 1001).reshape(
        -1, 1, 1, 1
    )
    p_at_c, _ = self._unnormalized_p(points)

    self.fac = jnp.max(p_at_c / self.simple.prob(loc_c), axis=0)
    jax.debug.print("🎲 CFG {fac}", fac=self.fac.mean())

  def _unnormalized_p(self, x):
    w = self.w
    logp_cfg = (1 + w) * self.pdf_c.log_prob(x) - w * self.pdf_u.log_prob(x)
    return jnp.exp(logp_cfg), logp_cfg

  def rejection_sample(
      self,
      seed: jax.Array,
      max_samples: int = 1_000,
  ) -> tuple[jax.Array, jax.Array]:
    """Rejection sampling, try `max_samples`, take first match."""
    rng_sample, rng_uni = jax.random.split(seed, 2)
    # Shape (max_samples, b, seq_len, c)
    xs = self.simple.sample(seed=rng_sample, sample_shape=(max_samples,))
    facq = self.fac * self.simple.prob(xs)
    ys = jax.random.uniform(rng_uni, shape=facq.shape, minval=0.0, maxval=facq)
    # Shape (max_samples, b, seq_len, c), True where `xs` is a valid sample
    # from p. We might have anywhere between 0 and `max_samples` valid samples!
    p, _ = self._unnormalized_p(xs)
    mask = ys < p
    # Now we need to do fancy tricks to get the first element in `mask` that is
    # True. We do this by making a shifted mask that is False for every element
    # after the first True.
    # > Example:
    # mask            [0, 1, 0, 1, 0, 0, 1, 0]
    # > implies:
    # cmask           [0, 1, 1, 1, 1, 1, 1, 1]
    # shifted_cmask   [0, 0, 1, 1, 1, 1, 1, 1]
    # keep            [0, 1, 0, 0, 0, 0, 0, 0]  # <- picks the first valid!
    cmask = jnp.cumsum(mask, axis=0).astype(jnp.bool_)
    shifted_cmask = jnp.pad(
        cmask, [(1, 0), (0, 0), (0, 0), (0, 0)], constant_values=False
    )[:-1]
    assert shifted_cmask.shape == mask.shape
    keep = jnp.logical_and(cmask, jnp.logical_not(shifted_cmask))
    # Now we can grab the first valid sample by doing a sum over the
    # `max_samples` dimension.
    sample = jnp.where(keep, xs, 0).sum(0)
    # If the rejection sampler fails, we fall back to the conditional
    # distribution.
    ok = mask.sum(0) > 0  # Shape (b, seq_len, c)
    # jax.debug.print("🎲 CFG ok {ok}%", ok=ok.mean() * 100)
    sample = jnp.where(
        ok, sample, self.pdf_c.sample(seed=rng_sample)
    )
    return sample, ok.mean() * 100

  def sample(
      self,
      seed: jax.Array,
      max_samples: int = 1_000,
  ) -> jax.Array:
    result, ok = self.rejection_sample(seed, max_samples)
    jax.debug.print("Debug ok={ok}%", ok=ok)
    return result

  # Unnormalized! But we only use it for ordering.
  def prob(self, xs: jax.Array) -> jax.Array:
    p, _ = self._unnormalized_p(xs)
    return p

  def log_prob(self, xs: jax.Array) -> jax.Array:
    _, lp = self._unnormalized_p(xs)
    return lp


def decode_masked(
    rng: jax.Array,
    labels: jax.Array,
    seq_len: int,
    feature_dim: int,
    model: givt.Model,
    variables: flax.core.FrozenDict,
    config: MaskedGenerationConfig,
) -> DecodeState:
  """Implements an masked bidirectional sampling loop.

  This function implements the loop from the docstring.

  Args:
    rng: RNG, only required if sampling.
    labels: Shape (b,), labels per batch. Determines batch size.
    seq_len: How many tokens to sample per batch.
    feature_dim: Output dimension of the VAE, i.e., number of channels, `c`.
    model: GIVT model to sample from.
    variables: Variables of the model.
    config: Configures style.

  Returns:
    Final state.
  """
  logging.info("Masked Generation Config:\n%s", config)

  if model.style != "masked":
    raise ValueError(f"Need masked model! Got `{model.style}`.")

  (b,) = labels.shape
  all_masked_input = jnp.zeros((b, seq_len, feature_dim))
  init_state = DecodeState.make(
      rng,
      all_masked_input,
      num_logits=model.num_logits,
      num_steps=config.num_steps,
  )

  def loop_cond_fn(state: DecodeState):
    return state.step < state.num_steps

  def tokens_to_logits(tokens, input_mask, drop_labels=None):
    return model.apply(
        variables,
        tokens,
        labels=labels,
        # Note that the model applies the mask token internally given the input.
        input_mask=input_mask,
        drop_labels=drop_labels,
        method="decode",
    )

  def loop_body_fn(state: DecodeState) -> DecodeState:
    # 1 where we should mask, cumulative.
    unknown = jnp.logical_not(state.total_uncovered)

    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = (state.step + 1) / config.num_steps
    # Note that the mask schedule inverts the function, so `mask_ratio` givts
    # near 1 and goes to 0 monotonically.
    mask_ratio = givt.apply_mask_schedule(ratio, method=config.schedule)
    mask_len = jnp.floor(seq_len * mask_ratio).reshape(1, 1)
    num_unknown = jnp.sum(unknown, axis=-1, keepdims=True)
    mask_len = jnp.maximum(
        0,
        # Keeps at least one of prediction in this round: Avoids the case where
        # mask_len is equal to num_unknown, in which case the mask is not
        # updated! We substract 1 to always remove at least one masked token.
        jnp.minimum(num_unknown - 1, mask_len))

    # Run model ---
    logits = tokens_to_logits(state.current_inputs_q, unknown)
    # Book keeping: store all logits.
    state = state.set_logits_at_current_step(logits)

    pdf = model.get_pdf(logits)
    state, sample_rng = state.split_rng()
    if config.cfg_inference_weight > 0:
      drop_all_labels = jnp.full((b,), True, jnp.bool_)
      logits_uncond = tokens_to_logits(
          state.current_inputs_q, unknown, drop_labels=drop_all_labels
      )
      state = state.set_uncond_logits_at_current_step(logits_uncond)
      pdf_uncond = model.get_pdf(logits_uncond)
      state, cfg_rng = state.split_rng()
      pdf = CFGDensity(
          pdf_c=pdf,
          pdf_u=pdf_uncond,
          w=config.cfg_inference_weight,
          rng=cfg_rng,
      )
      sample, rejection_sampling_success = pdf.rejection_sample(sample_rng)
      state = state.set_rejection_sampling_success_at_current_step(
          rejection_sampling_success
      )
    else:
      sample = pdf.sample(seed=sample_rng)

    # Sample at the unknown spots.
    sampled = jnp.where(unknown[:, :, None], sample, state.current_inputs_q)
    assert sampled.shape == (b, seq_len, feature_dim), (
        sampled.shape,
        b,
        seq_len,
        feature_dim,
    )

    prob = pdf.prob(sampled)
    if model.multivariate:
      assert prob.ndim == 2  # (b, seq_len) already
    elif model.per_channel_mixtures or config.cfg_inference_weight > 0:
      # Independence accross channels.
      # This reduction is also required when using CFG and also
      # `model.per_channel_mixtures == False` due to the 2-step CFG redefining
      # the pdf, but the reduction is not needed without CFG.
      prob = prob.prod(-1)
    state = state.set_prob_at_current_step(prob)

    if config.ordering == "maskgit":
      ordering = jnp.where(unknown, prob, _CONFIDENCE_OF_KNOWN_TOKENS)
    else:
      raise NotImplementedError(config.ordering)

    assert ordering.shape == (b, seq_len), (ordering.shape, b, seq_len)

    temp = config.choice_temperature
    if config.should_anneal_temperature:
      temp *= (1. - ratio)

    # True where we should mask input. Note that this is cumulative (ie this
    # starts with all True and keeps getting more False entries as we go through
    # the steps).
    state, choice_rng = state.split_rng()
    masking = mask_by_random_topk(choice_rng, mask_len, ordering, temp)
    assert masking.shape == (b, seq_len)
    masking = jnp.where(mask_len == 0, jnp.zeros_like(masking), masking)

    # Remove the masked tokens from the sampled array for safety (the model will
    # again apply the mask anyway...).
    sampled = jnp.where(masking[:, :, None], jnp.zeros_like(sampled), sampled)

    # Get next_uncover ---
    # New tokens to uncover (non cumulative): where it was unknown
    # but is now known.
    next_uncover = jnp.logical_and(unknown, jnp.logical_not(masking))
    assert next_uncover.shape == (b, seq_len), (next_uncover.shape, b, seq_len)
    state = state.set_uncover_at_current_step(next_uncover)
    state = state.set_next_input(sampled)
    return state.increment_step()

  return jax.lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
