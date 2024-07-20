"""Model definition for GPT-2."""
from dataclasses import dataclass

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from utils import recover_tree


@dataclass
class GPTConfig:
  vocab_size: int = 50257
  block_size: int = 1024
  emb_dim: int = 768
  num_heads: int = 12
  num_layers: int = 12


class SelfAttention(nn.Module):
  num_heads: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    x = nn.Dense(3 * x.shape[-1], dtype=self.dtype, name="c_attn")(x)
    q, k, v = jnp.split(x, 3, axis=-1)

    # Partition heads.
    q = einops.rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
    k = einops.rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
    v = einops.rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

    scale = 1.0 / jnp.sqrt(k.shape[-1])
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    attn = jnp.where(mask, attn, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(attn, axis=-1)  # (b, h, t, t)

    x = attn @ v  # (b, h, t, t) @ (b, h, t, d) -> (b, h, t, d)
    x = einops.rearrange(x, "b h t d -> b t (h d)")
    out = nn.Dense(x.shape[-1], dtype=self.dtype, name="c_proj")(x)
    return out


class MlpBlock(nn.Module):
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    out_dim = x.shape[-1]
    x = nn.Dense(4 * out_dim, name="c_fc")(x)
    x = nn.gelu(x)
    x = nn.Dense(out_dim, name="c_proj")(x)
    return x


class Block(nn.Module):
  emb_dim: int
  num_heads: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    attn = SelfAttention(self.num_heads, dtype=self.dtype, name="attn")
    mlp = MlpBlock(dtype=self.dtype, name="mlp")
    x = x + attn(nn.LayerNorm(name="ln_1")(x), mask)
    x = x + mlp(nn.LayerNorm(name="ln_2")(x))
    return x


class Transformer(nn.Module):
  emb_dim: int
  num_heads: int
  num_layers: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    for i in range(self.num_layers):
      x = Block(self.emb_dim, self.num_heads, dtype=self.dtype, name=str(i))(x, mask)
    return x


class GPT(nn.Module):
  cfg: GPTConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    _, T = x.shape
    assert T <= self.cfg.block_size, f"Input sequence length {T} is greater than block size {self.cfg.block_size}"

    causal_mask = nn.make_causal_mask(x)

    wte = nn.Embed(
        self.cfg.vocab_size,
        features=self.cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=0.02),
        name="wte")
    wpe = nn.Embed(
        self.cfg.block_size,
        features=self.cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=0.02),
        name="wpe")

    tok_emb = wte(x)
    pos_emb = wpe(jnp.arange(T, dtype=jnp.int32))

    x = tok_emb + pos_emb

    # Apply transformer blocks.
    x = Transformer(
        self.cfg.emb_dim, self.cfg.num_heads, self.cfg.num_layers, name="h")(
            x, mask=causal_mask)

    # Final layer norm and classification.
    x = nn.LayerNorm(name="ln_f")(x)
    x = wte.attend(x)
    return x


def get_config(variant: str):
  assert variant in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
  config_args = {
      "gpt2": dict(num_layers=12, num_heads=12, emb_dim=768),  # 124M
      "gpt2-medium": dict(num_layers=24, num_heads=16, emb_dim=1024),  # 350M
      "gpt2-large": dict(num_layers=36, num_heads=20, emb_dim=1280),  # 774M
      "gpt2-xl": dict(num_layers=48, num_heads=25, emb_dim=1600),  # 1558M
  }[variant]
  return GPTConfig(**config_args)


def load_hf_pretrained(variant: str):
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
