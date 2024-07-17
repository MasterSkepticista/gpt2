"""Trains GPT-2."""
import functools
import os
from dataclasses import dataclass
from clu.parameter_overview import get_parameter_overview

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

lead_host = jax.process_index() == 0
if os.environ.get("OMPI_COMM_WORLD_SIZE", -1) != -1:
  jax.distributed.initialize()


def info(*a, **k):
  if lead_host:
    print(*a, **k)


@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50257
  emb_dim: int = 768
  num_heads: int = 12
  num_layers: int = 12

class CausalSelfAttention(nn.Module):
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    _, T, C = x.shape
    x = nn.Dense(3 * x.shape[-1], dtype=self.dtype, name="c_attn")(x)
    q, k, v = jnp.split(x, 3, axis=-1)

    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    mask = jnp.where(mask == 0, -1e10, 0)
    attn = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(C) + mask
    attn = nn.softmax(attn, axis=-1)

    x = attn @ v
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
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    csa = CausalSelfAttention(dtype=self.dtype, name="attn")
    mlp = MlpBlock(dtype=self.dtype, name="mlp")
    x = x + csa(nn.LayerNorm(name="ln_1")(x))
    x = x + mlp(nn.LayerNorm(name="ln_2")(x))
    return x

class Transformer(nn.Module):
  emb_dim: int
  num_heads: int
  num_layers: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    for i in range(self.num_layers):
      x = Block(self.emb_dim, self.num_heads, dtype=self.dtype, name=str(i))(x)
    return x

class GPT(nn.Module):
  cfg: GPTConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    _, T = x.shape
    assert T <= self.cfg.block_size, f"Input sequence length {T} is greater than block size {self.cfg.block_size}"

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
    pos_emb = wpe(jnp.arange(T, dtype=jnp.int32))[None, :]

    x = tok_emb + pos_emb

    # Apply transformer blocks.
    x = Transformer(self.cfg.emb_dim, self.cfg.num_heads, self.cfg.num_layers, name="h")(x)
    
    # Final layer norm and classification.
    x = nn.LayerNorm(name="ln_f")(x)
    x = x @ wte.embedding.transpose(1, 0)
    return x


config = GPTConfig()
model = GPT(config)
variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1), dtype=jnp.int32))
print(get_parameter_overview(variables))
apply_fn = functools.partial(model.apply, variables)
print(jax.jit(apply_fn).lower(jnp.ones((1, 1024), dtype=jnp.int32)).cost_analysis()["flops"] / 2e9)