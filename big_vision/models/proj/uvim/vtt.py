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

"""Simple vision-text transformer with encoder-decoder architecture.

Used abbreviations for dimension annotations:
  B: batch size.
  H: image height.
  W: image width.
  P: number of patches (PH/PW: number of patches in height/width dimensions).
  E: embedding size.
  L: sequence length of text tokens.
  V: vocab size.
"""
from typing import Sequence
from big_vision import utils
from big_vision.models import common
from big_vision.models import vit
import einops
import flax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np


def shift_right(x, axis=1):
  """Shift to the right on given axis with padding value 0."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(x, pad_widths, constant_values=0)
  return padded[:, :-1]


class EncoderDecoderBlock(nn.Module):
  """Transformer encoder-decoder layer."""
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.
  decode: bool = False

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
    # Decoder block.
    x = nn.LayerNorm(name="LayerNorm1")(targets)
    x = nn.SelfAttention(
        num_heads=self.num_heads, use_bias=False, broadcast_dropout=False,
        dropout_rate=self.dropout_rate, decode=self.decode, name="SelfAttn")(
            x, decoder_mask, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(name="LayerNorm2")(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads, use_bias=False, broadcast_dropout=False,
        dropout_rate=self.dropout_rate, name="CrossAttn")(
            y, encoded, deterministic=deterministic)
    y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(name="LayerNorm3")(y)
    z = vit.MlpBlock(mlp_dim=self.mlp_dim, dropout=self.dropout_rate,
                     name="MLP")(z, deterministic=deterministic)

    return y + z


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation."""
  emb_dim: int
  mlp_dim: int
  num_heads: int
  num_layers: int
  dropout_rate: float = 0.
  output_vocab_size: int = 32000
  zero_decoder_seq: bool = False

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
      y = shift_right(y)
    y = nn.Embed(self.output_vocab_size, self.emb_dim, name="EmbedTargets",
                 embedding_init=nn.initializers.normal(stddev=1.0))(y)
    if self.zero_decoder_seq:
      y = jnp.zeros_like(y)
    y = common.AddPositionEmbs(
        decode=decode, name="PosEmbedTargets")(y, pos_emb)
    y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

    for lyr in range(self.num_layers):
      y = EncoderDecoderBlock(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate, decode=decode,
          name=f"EncDecBlock{lyr}")(y, encoded, decoder_mask=decoder_mask,
                                    deterministic=deterministic)
    y = nn.LayerNorm(name="LayerNorm")(y)
    logits = nn.Dense(self.output_vocab_size, kernel_init=nn.initializers.zeros,
                      name="LogitsDense")(y)
    return logits


class Model(nn.Module):
  """Transformer Model for sequence to sequence translation."""
  patches: ml_collections.ConfigDict
  # Encoder/decoder shared params:
  num_heads: int = 8
  num_layers: int = 6
  mlp_dim: int = 2048
  dropout_rate: float = 0.
  # Decoder params:
  emb_dim: int = 512
  vocab_size: int = 32000
  seq_len: int = 256
  # Encoder params:
  input_size: Sequence[int] = (256, 256)
  posemb_type: str = "sincos2d"  # Can also be "learn"
  zero_decoder_seq: bool = False

  def setup(self):
    grid_size = np.array(self.input_size) // np.array(self.patches.size)
    self.pos_emb_for_encoder = vit.get_posemb(
        self, self.posemb_type, grid_size, self.emb_dim,
        "pos_embedding_encoder")
    self.pos_emb_for_decoder = vit.get_posemb(
        self, self.posemb_type, (1, self.seq_len), self.emb_dim,
        "pos_embedding_decoder")

    self.encoder = vit.Encoder(
        depth=self.num_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout_rate)
    self.decoder = Decoder(
        num_layers=self.num_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        emb_dim=self.emb_dim,
        output_vocab_size=self.vocab_size,
        zero_decoder_seq=self.zero_decoder_seq,
    )
    self.conv = nn.Conv(self.emb_dim, self.patches.size, padding="VALID",
                        strides=self.patches.size, name="EmbedPatches")

  def encode(self, image, train=False):
    """Encodes input image or embeddings."""
    emb = self.conv(image)
    patch_embeddings = einops.rearrange(emb, "B PH PW E -> B (PH PW) E")
    encoded, _ = self.encoder(
        patch_embeddings + self.pos_emb_for_encoder, deterministic=not train)
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

  def __call__(self, image, text, *, decode=False, train=False):
    """Applies Transformer model on the inputs.

    Args:
      image: batch of images [B, H, W, 3].
      text: batch of tokenized texts [B, L].
      decode: whether to prepare and use an autoregressive cache.
      train: whether it is training.

    Returns:
      logits array from full transformer [B, L, V].
    """
    encoded = self.encode(image, train=train)
    return self.decode(encoded, text, decode=decode, train=train)


def load(init_params, init_files, model_params=None,
         dont_load=("head/kernel", "head/bias", "cls")):
  """Loads params from init checkpoint and merges into init_params."""
  del model_params
  if isinstance(init_files, str):
    # A shortcut for a single file checkpoint of a vtt model.
    ckpt_params = utils.load_params(None, init_files)
    ckpt_params = flax.training.checkpoints.convert_pre_linen(ckpt_params)
    if init_params is not None:
      ckpt_params = common.merge_params(ckpt_params, init_params, dont_load)
  else:
    init_files = {**init_files}  # Shallow copy because we'll pop stuff off.

    enc_init = init_files.pop("encoder", None)
    if enc_init:
      ckpt_params = init_params.copy()
      vit_params = {
          "pos_embedding": ckpt_params["pos_embedding_encoder"],
          "Transformer": ckpt_params["encoder"],
          "embedding": ckpt_params["EmbedPatches"],
      }
      encoder_params = vit.load(
          vit_params, enc_init, model_cfg={},
          dont_load=dont_load)
      ckpt_params["encoder"] = encoder_params["Transformer"]
      ckpt_params["pos_embedding_encoder"] = encoder_params["pos_embedding"]
      ckpt_params["EmbedPatches"] = encoder_params["embedding"]
    else:
      raise ValueError("Only encoder init is supported: {}.".format(init_files))

  return ckpt_params
