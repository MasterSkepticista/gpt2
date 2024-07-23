"""Model definition for GPT-2."""
import functools
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from absl import flags
from flax import linen as nn
from utils import recover_tree

flags.DEFINE_bool(
    "flash_attention", default=True, help="Whether to use JAX SDPA kernel.")
FLAGS = flags.FLAGS


class SelfAttention(nn.Module):
  """Multi-Headed Causal Self-Attention."""
  num_heads: int
  implementation: Literal["xla", "cudnn"] = "xla"
  kernel_init: Callable[..., Any] = nn.initializers.normal(0.02)
  bias_init: Callable[..., Any] = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    assert x.ndim == 3, "Input must be of shape [batch, time, features]"
    assert x.shape[-1] % self.num_heads == 0, (
        f"Embedding dimension {x.shape[-1]} must be divisible by num_heads {self.num_heads}"
    )

    b, t, c = x.shape
    head_dim = c // self.num_heads

    dense = functools.partial(
        nn.DenseGeneral,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype)

    q, k, v = (dense(name=name)(x) for name in ("q", "k", "v"))

    if FLAGS.flash_attention:
      # Flash attention.
      x = jax.nn.dot_product_attention(
          q, k, v, is_causal=True, implementation=self.implementation)
    else:
      # Standard attention (for educational purposes).
      mask = nn.make_causal_mask(jnp.zeros((b, t)))
      scale = 1.0 / jnp.sqrt(c)
      attn = jnp.einsum("...qhd,...khd->...hqk", q, k * scale)
      attn = jnp.where(mask, attn, jnp.finfo(jnp.float32).min)
      attn = jax.nn.softmax(attn, axis=-1)
      x = jnp.einsum("...hqk,...khd->...qhd", attn, v)

    x = jnp.reshape(x, (b, t, c))
    out = nn.Dense(
        features=c,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name="c_proj",
        dtype=self.dtype)(x)  # yapf: disable
    return out


class MlpBlock(nn.Module):
  """MLP block."""
  kernel_init: Callable[..., Any] = nn.initializers.normal(0.02)
  bias_init: Callable[..., Any] = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    dense = functools.partial(
        nn.Dense,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype)
    out_dim = x.shape[-1]

    x = dense(4 * out_dim, name="c_fc")(x)
    x = nn.gelu(x)
    x = dense(
        out_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        name="c_proj")(x)  # yapf: disable
    return x


class Block(nn.Module):
  """Transformer block."""
  emb_dim: int
  num_heads: int
  sdpa_implementation: Literal["xla", "cudnn"]
  kernel_init: Callable[..., Any] = nn.initializers.normal(stddev=0.02)
  bias_init: Callable[..., Any] = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    attn = SelfAttention(
        self.num_heads,
        implementation=self.sdpa_implementation,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        name="attn")

    mlp = MlpBlock(
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        name="mlp")

    ln_1 = nn.LayerNorm(dtype=self.dtype, name="ln_1")
    ln_2 = nn.LayerNorm(dtype=self.dtype, name="ln_2")

    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))
    return x


class Transformer(nn.Module):
  emb_dim: int
  num_heads: int
  num_layers: int
  sdpa_implementation: Literal["xla", "cudnn"]
  kernel_init: Callable[..., Any] = nn.initializers.normal(stddev=0.02)
  bias_init: Callable[..., Any] = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    for i in range(self.num_layers):
      x = Block(
          self.emb_dim,
          self.num_heads,
          sdpa_implementation=self.sdpa_implementation,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name=str(i))(x)  # yapf: disable
    return x


class GPT(nn.Module):
  """GPT-2 architecture."""
  vocab_size: int
  block_size: int
  emb_dim: int
  num_heads: int
  num_layers: int
  sdpa_implementation: Literal["xla", "cudnn"] = "xla"
  embedding_init: Callable[..., Any] = nn.initializers.normal(stddev=0.02)
  kernel_init: Callable[..., Any] = nn.initializers.normal(stddev=0.02)
  bias_init: Callable[..., Any] = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    _, T = x.shape
    assert T <= self.block_size, (
        f"Input sequence length {T} is greater than block size {self.block_size}"
    )

    wte = nn.Embed(
        self.vocab_size,
        features=self.emb_dim,
        embedding_init=self.embedding_init,
        dtype=self.dtype,
        name="wte")
    wpe = nn.Embed(
        self.block_size,
        features=self.emb_dim,
        embedding_init=self.embedding_init,
        dtype=self.dtype,
        name="wpe")

    tok_emb = wte(x)
    pos_emb = wpe(jnp.arange(T, dtype=jnp.int32))

    x = tok_emb + pos_emb

    # Apply transformer blocks.
    x = Transformer(
        self.emb_dim,
        self.num_heads,
        self.num_layers,
        sdpa_implementation=self.sdpa_implementation,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        name="h")(x)  # yapf: disable

    # Final layer norm and classification.
    x = nn.LayerNorm(dtype=self.dtype, name="ln_f")(x)
    x = wte.attend(x)
    return x


def load_hf_pretrained(variant: str):
  # FIXME: Does not work with the current split-qkv dense matrices.
  """Load HF-Transformers GPT2 weights."""
  assert variant in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

  from transformers import GPT2LMHeadModel
  print("Loading pretrained weights: %s" % variant)
  hf_params = GPT2LMHeadModel.from_pretrained(variant).state_dict()
  hf_params = {k: jnp.asarray(v.numpy()) for k, v in hf_params.items()}

  # Rename torch params to flax params.
  hf_params = {k.replace("transformer.", ""): v for k, v in hf_params.items()}
  hf_params = {
      k.replace("wte.weight", "wte.embedding"): v for k, v in hf_params.items()
  }
  hf_params = {
      k.replace("wpe.weight", "wpe.embedding"): v for k, v in hf_params.items()
  }
  hf_params = {
      (k.replace(".weight", ".scale") if "ln" in k else k): v
      for k, v in hf_params.items()
  }
  hf_params = {k.replace(".weight", ".kernel"): v for k, v in hf_params.items()}
  hf_params.pop("lm_head.kernel")  # Same as wte.embedding

  # Convert to Flax nested tree format.
  names, values = zip(*hf_params.items())
  restored_params = recover_tree(names, values)
  return restored_params
