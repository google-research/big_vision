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

"""CNN encoder/decoder architecture based on the VQ-GAN and MaskGIT papers.

Adapted from https://github.com/google-research/maskgit/blob/main/maskgit/nets/vqgan_tokenizer.py. # pylint: disable=line-too-long
"""

import dataclasses
import functools
import math
from typing import Any, Sequence

from big_vision import utils
from big_vision.models import common
from big_vision.models.proj.givt import vae

import einops
import flax.linen as nn
import flax.training.checkpoints

import jax
import jax.numpy as jnp


def _get_norm_layer(train, dtype, norm_type="BN"):
  """Create normalization layers.

  Args:
    train: Whether to use the layer in training or inference mode.
    dtype: Layer output type.
    norm_type: Which normalization to use "BN", "LN", or "GN".

  Returns:
    An instance of the the layer.
  """
  if norm_type == "BN":
    return functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name=None,
        axis_index_groups=None,
        dtype=jnp.float32,
        use_fast_variance=False)
  elif norm_type == "LN":
    return functools.partial(nn.LayerNorm, dtype=dtype, use_fast_variance=False)
  elif norm_type == "GN":
    return functools.partial(nn.GroupNorm, dtype=dtype, use_fast_variance=False)
  else:
    raise NotImplementedError


def _tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.

  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
      (1,) + strides + (1,), padding)
  return pool_sum / pool_denom


def _upsample(x, factor=2, method="nearest"):
  n, h, w, c = x.shape
  x = jax.image.resize(x, (n, h * factor, w * factor, c), method=method)
  return x


def _dsample(x):
  return _tensorflow_style_avg_pooling(
      x, (2, 2), strides=(2, 2), padding="same")


def get_h_w_pixelshuffle(hw, pixel_shuffle_patch_size):
  # Compute h, w after space-to-depth transformation and before flattening,
  # assuming the imge before space-to-depth transformation was square.
  ph, pw = pixel_shuffle_patch_size
  s = int(math.sqrt(hw * ph * pw))
  h, w = s // ph, s // pw
  assert h * w == hw, f"Length {hw} incompatible with pixelshuffle ({ph}, {pw})"
  return h, w


class ResBlock(nn.Module):
  """Basic Residual Block."""
  filters: int
  norm_fn: Any
  conv_fn: Any
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu
  use_conv_shortcut: bool = False

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    input_dim = x.shape[-1]
    residual = x
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    if input_dim != self.filters:
      if self.use_conv_shortcut:
        residual = self.conv_fn(
            self.filters, kernel_size=(3, 3), use_bias=False)(
                x)
      else:
        residual = self.conv_fn(
            self.filters, kernel_size=(1, 1), use_bias=False)(
                x)
    return x + residual


class Encoder(nn.Module):
  """Encoder Blocks."""

  filters: int
  num_res_blocks: int
  channel_multipliers: list[int]
  embedding_dim: int
  conv_downsample: bool = False
  norm_type: str = "GN"
  activation_fn_str: str = "swish"
  dtype: int = jnp.float32

  def setup(self) -> None:
    if self.activation_fn_str == "relu":
      self.activation_fn = nn.relu
    elif self.activation_fn_str == "swish":
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    conv_fn = nn.Conv
    norm_fn = _get_norm_layer(
        train=train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    x = conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i < num_blocks - 1:
        if self.conv_downsample:
          x = conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
        else:
          x = _dsample(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.embedding_dim, kernel_size=(1, 1))(x)
    return x


class Decoder(nn.Module):
  """Decoder Blocks."""

  filters: int
  num_res_blocks: int
  channel_multipliers: list[int]
  norm_type: str = "GN"
  activation_fn_str: str = "swish"
  output_dim: int = 3
  dtype: Any = jnp.float32

  def setup(self) -> None:
    if self.activation_fn_str == "relu":
      self.activation_fn = nn.relu
    elif self.activation_fn_str == "swish":
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    conv_fn = nn.Conv
    norm_fn = _get_norm_layer(
        train=train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    num_blocks = len(self.channel_multipliers)
    filters = self.filters * self.channel_multipliers[-1]
    x = conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    for i in reversed(range(num_blocks)):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i > 0:
        x = _upsample(x, 2)
        x = conv_fn(filters, kernel_size=(3, 3))(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.output_dim, kernel_size=(3, 3))(x)
    return x


class Model(vae.Model):
  """CNN Model."""

  filters: int = 128
  num_res_blocks: int = 2
  channel_multipliers: list[int] = dataclasses.field(default_factory=list)
  conv_downsample: bool = False
  activation_fn: str = "swish"
  norm_type: str = "GN"
  output_dim: int = 3
  dtype: Any = jnp.float32
  # If True, rescale the input [-1, 1] -> [0, 1] and clip logvar to [-30, 20]
  malib_ckpt: bool = False
  pixel_shuffle_patch_size: tuple[int, int] = (1, 1)

  def setup(self) -> None:
    # Encoder and decoder
    self.encoder = Encoder(
        filters=self.filters,
        num_res_blocks=self.num_res_blocks,
        channel_multipliers=self.channel_multipliers,
        norm_type=self.norm_type,
        activation_fn_str=self.activation_fn,
        embedding_dim=2 * self.codeword_dim,
        conv_downsample=self.conv_downsample,
        dtype=self.dtype,
        name="cnn_encoder",
    )
    self.decoder = Decoder(
        filters=self.filters,
        num_res_blocks=self.num_res_blocks,
        channel_multipliers=self.channel_multipliers,
        norm_type=self.norm_type,
        activation_fn_str=self.activation_fn,
        output_dim=self.output_dim,
        dtype=self.dtype,
        name="cnn_decoder",
    )

  def _maybe_rescale_input(self, x):
    return (x + 1.0) / 2.0 if self.malib_ckpt else x

  def _maybe_rescale_output(self, x):
    return 2.0 * x - 1.0 if self.malib_ckpt else x

  def _maybe_clip_logvar(self, logvar):
    return jnp.clip(logvar, -30.0, 20.0) if self.malib_ckpt else logvar

  def encode(
      self,
      x: jax.Array,
      *,
      train: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    x = self._maybe_rescale_input(x)
    x = self.encoder(x, train=train)                  # (2, 16, 16, 64)
    assert x.shape[1] == x.shape[2], f"Square spatial dims. required: {x.shape}"
    mu, logvar = jnp.split(x, 2, axis=-1)             # (2, 16, 16, 32) x 2
    logvar = self._maybe_clip_logvar(logvar)

    def _space_to_depth(z):
      ph, pw = self.pixel_shuffle_patch_size
      return einops.rearrange(
          z, "b (h ph) (w pw) c -> b (h w) (c ph pw)",
          ph=ph, pw=pw
      )   # (2, 256 // (ph * pw), 64 * ph * pw)

    mu, logvar = _space_to_depth(mu), _space_to_depth(logvar)

    return mu, logvar

  def decode(self, x: jax.Array, train: bool = False) -> jax.Array:
    # Decode
    ph, pw = self.pixel_shuffle_patch_size
    h, w = get_h_w_pixelshuffle(x.shape[1], (ph, pw))

    x = einops.rearrange(
        x, "b (h w) (c ph pw) -> b (h ph) (w pw) c",
        h=h, w=w,
        ph=ph, pw=pw
    )  # (2, 16, 16, 32)
    x = self.decoder(x, train=train)   # (2, 256, 256, 3)
    x = self._maybe_rescale_output(x)
    x = jnp.clip(x, -1.0, 1.0)

    return x


def load(
    init_params: Any,
    init_file: str,
    model_params: Any = None,
    dont_load: Sequence[str] = (),
    malib_ckpt: bool = False,
    use_ema_params: bool = False,
) -> Any:
  """Loads params from init checkpoint and merges into init_params.

  Args:
    init_params: pytree with (previously initialized) model parameters.
    init_file: Path of the checkpoint to load.
    model_params: Dict containing the model config.
    dont_load: Sequence of (flattened) parameter names which should not be
      loaded.
    malib_ckpt: Whether the given init_file is a malib checkpoint.
    use_ema_params: Whether to load the EMA params (for malib checkpoints).

  Returns:
    pytree containing the loaded model parameters.
  """
  # `model_params` is unused here, but we still include it to conform with the
  # general big_vision interface, cf. the core models in big_vision/models/.
  del model_params

  assert malib_ckpt or (not use_ema_params), (
      "Loading EMA parameters is only supported for malib checkpoints.")

  if malib_ckpt:
    # Locally disable transfer guard since restore_checkpoint does not allow for
    # fine-grained sharding control.
    with jax.transfer_guard("allow"):
      vaegan_params = flax.training.checkpoints.restore_checkpoint(
          init_file, None)
    vaegan_params_flat = utils.tree_flatten_with_names(vaegan_params)[0]
    prefix_old = "ema_params/" if use_ema_params else "g_params/"
    vaegan_params_flat = [(k.replace(prefix_old, "cnn_"), v)
                          for k, v in vaegan_params_flat if prefix_old in k]
    params = utils.tree_unflatten(vaegan_params_flat)
  else:
    params = flax.core.unfreeze(utils.load_params(init_file))

  if init_params is not None:
    params = common.merge_params(params, init_params, dont_load)
  return params
