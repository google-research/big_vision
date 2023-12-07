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

"""Model definitions for CapPa (https://arxiv.org/abs/2306.07915).

Used abbreviations for dimension annotations:
  B: batch size.
  H: image height.
  W: image width.
  P: number of patches (PH/PW: number of patches in height/width dimensions).
  E: embedding size.
  L: sequence length of text tokens.
  V: vocab size.
"""

from collections.abc import Sequence

from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
import flax
import flax.linen as nn
from flax.linen import partitioning
import jax
import jax.numpy as jnp


def shift_right(x, axis=1, constant_values=0):
  """Shift to the right on given axis with padding value 0."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(x, pad_widths, constant_values=constant_values)
  # Cuts off the rightmost slice of size along the `axis` dimension.
  # Note that `list[:-1]`` is the same as `list[slice(-1)]`.
  return padded[tuple(slice(-1 if i == axis else None) for i in range(x.ndim))]


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block with option to deactivate bias."""
  mlp_dim: int | None = None  # Defaults to 4x input dim
  dropout: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, use_bias=self.use_bias, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, use_bias=self.use_bias, **inits)(x)
    return x


class EncoderDecoderBlock(nn.Module):
  """Transformer encoder-decoder layer."""
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.
  decode: bool = False
  use_bias: bool = True

  @nn.compact
  def __call__(self, targets, encoded, decoder_mask=None, deterministic=True):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: target text embeddings [B, L, E].
      encoded: encoded image patches from encoder [B, P, E].
      decoder_mask: decoder self-attention mask.
      deterministic: bool, deterministic or not (to apply dropout).

    Returns:
      output after transformer encoder-decoder block [B, L, E].
    """
    def wlc(f):
      dim_names = ("act_batch", "act_len", "act_emb")
      return nn.with_logical_constraint(f, dim_names)

    # Decoder block.
    x = wlc(nn.LayerNorm(name="LayerNorm1", use_bias=self.use_bias)(targets))
    x = wlc(nn.SelfAttention(
        num_heads=self.num_heads, use_bias=False, broadcast_dropout=False,
        dropout_rate=self.dropout_rate, decode=self.decode, name="SelfAttn")(
            x, decoder_mask, deterministic=deterministic))
    x = wlc(nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic))
    x = wlc(x + targets)

    if encoded is not None:
      # Encoder-Decoder block.
      y = wlc(nn.LayerNorm(name="LayerNorm2", use_bias=self.use_bias)(x))
      y = wlc(nn.MultiHeadDotProductAttention(
          num_heads=self.num_heads, use_bias=False, broadcast_dropout=False,
          dropout_rate=self.dropout_rate, name="CrossAttn")(
              y, encoded, deterministic=deterministic))
      y = wlc(
          nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic))
      y = wlc(y + x)
    else:
      y = x

    # MLP block.
    z = wlc(nn.LayerNorm(name="LayerNorm3", use_bias=self.use_bias)(y))
    z = wlc(MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout_rate, use_bias=self.use_bias,
        name="MLP")(z, deterministic=deterministic))

    return wlc(y + z), None


class Decoder(nn.Module):
  """Transformer decoder with parallel prediction."""
  emb_dim: int
  mlp_dim: int
  num_heads: int
  num_layers: int
  dropout_rate: float = 0.
  output_vocab_size: int = 32_000

  # Masked prediction training mode
  masked_pred_prob: float = 0.
  masking_ratio: float = 0.

  # Whether to use bias in MLP blocks and LN
  use_bias: bool = True

  scan: bool = False
  remat_policy: str = "nothing_saveable"

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               pos_emb,
               decoder_mask=None,
               decode=False,
               deterministic=True,
               max_decode_length=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded image patches from encoder [B, P, E].
      targets: target text tokens [B, L].
      pos_emb: positional embeddings.
      decoder_mask: decoder self-attention mask.
      decode: bool, whether to perform fast autoregressive decoding with cache.
      deterministic: bool, deterministic or not (to apply dropout).
      max_decode_length: optional max length for positional embeddings.

    Returns:
      output of a transformer decoder [B, L, V].
    """
    y = targets.astype("int32")
    if not decode:
      if self.masked_pred_prob > 0.0 and not deterministic:
        # Binary random variable indicating whether to do masked prediction

        def _add_random_masks(a):
          # Generate random mask
          n_masked = int(self.masking_ratio * a.shape[1])
          mask_locations = jnp.zeros(a.shape[:2], dtype=jnp.int32)
          mask_locations = mask_locations.at[:, :n_masked].set(1)
          mask_locations = jax.random.permutation(
              self.make_rng("dropout"), mask_locations, axis=1, independent=True
          )
          # Replace mask locations with mask token index (=vocab_size)
          a_masked = jnp.where(mask_locations, self.output_vocab_size, a)
          return a_masked

        def where(mask, x, y):
          mask = mask.reshape((-1,) + (1,) * (x.ndim - 1))
          return jnp.where(mask, x, y)

        do_masked_pred = (
            jax.random.uniform(self.make_rng("dropout"), (len(y),))
            < self.masked_pred_prob
        )
        y = where(do_masked_pred, _add_random_masks(y), shift_right(y))
        decoder_mask = where(
            do_masked_pred, jnp.ones_like(decoder_mask), decoder_mask
        )

      else:
        y = shift_right(y)

    embed = nn.Embed(
        self.output_vocab_size + (1 if self.masked_pred_prob > 0.0 else 0),
        self.emb_dim,
        name="EmbedTargets",
        embedding_init=nn.initializers.normal(stddev=1.0),
    )
    y = embed(y)

    y = common.AddPositionEmbs(
        decode=decode, name="PosEmbedTargets")(y, pos_emb)
    # NOTE: One could apply dropout on the decoder's inputs here. Whether to do
    # it or not, and if so, what is the best/common way, is to be determined.
    # y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

    if self.scan:
      # Mostly followed
      # https://github.com/google/maxtext/blob/4d99e30b3e0e0cb1d1aa11c7db7fffe18e301498/MaxText/layers.py#L1126
      # for the scanned version.
      # 1. remat
      enc_dec_block_remat = nn.remat(
          EncoderDecoderBlock,
          prevent_cse=False,
          static_argnums=(-1,),
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None))
      # 2. scan
      initializing = self.is_mutable_collection("params")
      param_scan_axis = 1
      params_spec = (param_scan_axis if initializing
                     else partitioning.ScanIn(param_scan_axis))
      dec_scanned = nn.scan(enc_dec_block_remat,
                            variable_axes={
                                "params": params_spec,
                                "cache": 0,
                            },
                            split_rngs={"params": True, "dropout": True},
                            in_axes=nn.broadcast,
                            length=self.num_layers)
      # 3. fprop
      y, _ = dec_scanned(num_heads=self.num_heads, mlp_dim=self.mlp_dim,
                         dropout_rate=self.dropout_rate, decode=decode,
                         use_bias=self.use_bias, name="EncDecBlock")(
                             y, encoded, decoder_mask, deterministic)
    else:
      for lyr in range(self.num_layers):
        y, _ = EncoderDecoderBlock(
            num_heads=self.num_heads, mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate, decode=decode,
            use_bias=self.use_bias, name=f"EncDecBlock{lyr}")(
                y, encoded, decoder_mask=decoder_mask,
                deterministic=deterministic)

    y = nn.LayerNorm(name="LayerNorm")(y)

    logits = nn.Dense(
        self.output_vocab_size,
        kernel_init=nn.initializers.zeros,
        name="LogitsDense",
    )(y)
    return logits


class Model(nn.Module):
  """Transformer Model for sequence to sequence translation."""
  # Encoder/decoder:
  num_heads: int = 8
  num_layers: int = 6
  mlp_dim: int = 2048
  emb_dim: int = 512
  enc_dropout_rate: float = 0.
  vocab_size: int = 32_000
  seq_len: int = 256

  # Encoder:
  patches: Sequence[int] = (16, 16)
  input_seq_len: int = 768
  posemb_type: str = "learn"
  patch_dropout: float = 0.

  # Decoder:
  decoder_num_heads: int = 0
  decoder_num_layers: int = 0
  decoder_mlp_dim: int = 0
  decoder_emb_dim: int = 0
  dec_dropout_rate: float = 0.
  # Probability of masked prediction rather than autoregressive prediciton.
  masked_pred_prob: float = 0.
  # Masking ratio for masked prediction.
  masking_ratio: float = 0.
  # Whether to use bias in decoder MLP blocks and LN.
  decoder_bias: bool = True

  scan: bool = False
  remat_policy: str = "nothing_saveable"

  def setup(self):

    self.encoder = vit.Model(
        patch_size=self.patches,
        width=self.emb_dim,
        depth=self.num_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.enc_dropout_rate,
        posemb=self.posemb_type,
        scan=self.scan,
        remat_policy=self.remat_policy,
    )

    self.pos_emb_for_decoder = vit.get_posemb(
        self,
        self.posemb_type,
        (1, self.seq_len),
        self.decoder_emb_dim or self.emb_dim,
        "pos_embedding_decoder",
    )
    self.decoder = Decoder(
        num_layers=self.decoder_num_layers or self.num_layers,
        mlp_dim=self.decoder_mlp_dim or self.mlp_dim,
        num_heads=self.decoder_num_heads or self.num_heads,
        dropout_rate=self.dec_dropout_rate,
        emb_dim=self.decoder_emb_dim or self.emb_dim,
        output_vocab_size=self.vocab_size,
        masked_pred_prob=self.masked_pred_prob,
        masking_ratio=self.masking_ratio,
        use_bias=self.decoder_bias,
        scan=self.scan,
        remat_policy=self.remat_policy,
    )

  def encode(self, image, train=False, return_enc_features=False):
    """Encodes input image or embeddings."""

    _, out = self.encoder(image, train=train)
    encoded = out["encoded"]

    # Return intermediate features if required
    if return_enc_features:
      return encoded, out

    return encoded

  def decode(self, encoded, targets, decode=False, train=False,
             max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded image patches from encoder [B, P, E].
      targets: target text tokens [B, L].
      decode: whether to prepare and use an autoregressive cache.
      train: whether it is training.
      max_decode_length: optional max length for positional embeddings.

    Returns:
      logits array from transformer decoder [B, L, V].
    """
    decoder_mask = None if decode else nn.make_causal_mask(targets)
    logits = self.decoder(
        encoded,
        targets,
        pos_emb=self.pos_emb_for_decoder,
        decoder_mask=decoder_mask,
        decode=decode,
        deterministic=not train,
        max_decode_length=max_decode_length)
    return logits

  def __call__(self, image, text, *, decode=False,
               train=False, return_enc_features=False):
    """Applies Transformer model on the inputs.

    Args:
      image: batch of images [B, H, W, 3].
      text: batch of tokenized texts [B, L].
      decode: whether to prepare and use an autoregressive cache.
      train: whether it is training.
      return_enc_features: whether to return the encoder features.

    Returns:
      logits array from full transformer [B, L, V].
    """
    if return_enc_features:
      encoded, out = self.encode(image, train=train, return_enc_features=True)
      return encoded, out

    encoded = self.encode(image, train=train)

    decoded = self.decode(encoded, text, decode=decode, train=train)
    return decoded


def load(init_params, init_files, model_params=None,
         dont_load=("head/kernel", "head/bias", "cls")):
  """Loads params from init checkpoint and merges into init_params."""

  if isinstance(init_files, str):
    # A shortcut for a single file checkpoint of a vtt model.
    ckpt_params = utils.load_params(init_files)
    ckpt_params = flax.training.checkpoints.convert_pre_linen(ckpt_params)
    ckpt_params = common.merge_params(ckpt_params, init_params, dont_load)

    # Detect attempts to load non-scan checkpoint into scan model if possible.
    if (model_params.get("scan") and
        "encoderblock" not in ckpt_params["encoder"]["Transformer"]):
      raise NotImplementedError("Loading a non-scan checkpoint into a "
                                "scan model is not supported yet!")
    if (not model_params.get("scan")
        and "encoderblock" in ckpt_params["encoder"]["Transformer"]):
      assert "decoder.*" in dont_load or "decoder/.*" in dont_load, (
          "Converting scan decoder to a non-scan one is not supported yet!")
      ckpt_params["encoder"] = utils.jit_cpu()(
          vit.scan_to_pyloop)(ckpt_params["encoder"])

  else:
    assert set(init_files) == {"encoder"}, "Only encoder init supported"
    enc_init = init_files["encoder"]
    ckpt_params = flax.core.freeze(init_params).unfreeze()
    vit_params = ckpt_params["encoder"]
    encoder_params = vit.load(
        vit_params, enc_init, model_cfg={},
        dont_load=dont_load)
    ckpt_params["encoder"] = encoder_params

  ckpt_params["encoder"]["pos_embedding"] = vit.resample_posemb(
      old=ckpt_params["encoder"]["pos_embedding"],
      new=init_params["encoder"]["pos_embedding"])

  return ckpt_params
