"""Flash Attention Pallas Benchmark."""
from absl import app

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from attention import naive_attention, cudnn_attention, flash_attention

def online_softmax(x: jax.Array):
  out = jnp.zeros_like(x).astype(jnp.float32)
  TILE_SIZE = 3

  m, s = -jnp.inf, 0
  for i in range(x.shape[0] // TILE_SIZE):
    idx = pl.dslice(i * TILE_SIZE, TILE_SIZE)
    x_tile = x[idx]
    
    m_i = jnp.max(x_tile)
    m_prev = m
    m = jnp.maximum(m, m_i)
    s = s * jnp.exp(m_prev - m) + jnp.sum(jnp.exp(x_tile - m))
  
  for i in range(x.shape[0] // TILE_SIZE):
    idx = pl.dslice(i * TILE_SIZE, TILE_SIZE)
    x_tile = x[idx]
    out = out.at[idx].set(jnp.exp(x_tile - m) / s)

  return out

def main(argv):
  B, T, H, C = 1, 1024, 12, 64
  keys = jax.random.split(jax.random.key(42), 4)

  # FA cuDNN works on f16 and bf16 only.
  q = jax.random.normal(keys[0], (B, T, H, C), jnp.bfloat16)
  k = jax.random.normal(keys[1], (B, T, H, C), jnp.bfloat16)
  v = jax.random.normal(keys[2], (B, T, H, C), jnp.bfloat16)
  do = jax.random.normal(keys[3], (B, T, H, C), jnp.bfloat16)

  # Forward pass
  o_ref = naive_attention(q, k, v, causal=True)
  o_flash = flash_attention(q, k, v, causal=True)
  print("Forward pass result match:", jnp.allclose(o_ref, o_flash, atol=1e-2, rtol=1e-2))

  # Backward pass
  def loss_ref(q, k, v):
    return jnp.sum(naive_attention(q, k, v, causal=True) * do)
  dq_ref, dk_ref, dv_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
  print("Reference shapes:", dq_ref.shape, dk_ref.shape, dv_ref.shape)

  def loss(q, k, v):
    return jnp.sum(flash_attention(q, k, v, causal=True) * do)
  dq_flash, dk_flash, dv_flash = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
  print("Flash shapes:", dq_flash.shape, dk_flash.shape, dv_flash.shape)

  print("Backward pass dQ match:", jnp.allclose(dq_ref, dq_flash, atol=1e-2, rtol=1e-2))
  print("Backward pass dK match:", jnp.allclose(dk_ref, dk_flash, atol=1e-2, rtol=1e-2))
  print("Backward pass dV match:", jnp.allclose(dv_ref, dv_flash, atol=1e-2, rtol=1e-2))






if __name__ == "__main__":
  app.run(main)