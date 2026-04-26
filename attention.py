"""Dot-product Attention Kernels."""
from functools import partial
import math

import jax
import jax.numpy as jnp

from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

@jax.jit
def naive_attention(
  query: jax.Array, key: jax.Array, value: jax.Array, causal: bool = False) -> jax.Array:
  """Naive dot-product-attention kernel that materializes everything into global memory.

  
  Args:
    query: jax.Array of shape (batch..., q_length, num_heads, depth)
    key: jax.Array of shape (batch..., kv_length, num_heads, depth)
    value: jax.Array of shape (batch..., kv_length, num_heads, depth)
    causal: bool, whether to apply causal masking.
  
  Returns:
    jax.Array of shape (batch..., q_length, num_heads, depth).
  """
  depth = query.shape[-1]
  attn_weights = jnp.einsum(
    "...qhd,...khd->...hqk", query, key) / jnp.sqrt(depth)
  
  if causal is not None:
    mask = jnp.tril(jnp.ones(attn_weights.shape[-2:], dtype=bool))
    attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
  attn_weights = jax.nn.softmax(attn_weights, axis=-1)
  out = jnp.einsum(
    "...hqk,...khd->...qhd", attn_weights, value)
  return out

def cudnn_flash_attention(
  query: jax.Array, key: jax.Array, value: jax.Array, causal: bool = False) -> jax.Array:
  """JAX built-in flash-attention cuDNN implementation.
  
  Args:
    query: jax.Array of shape (..., H, C) and bf16/f16 dtype.
    key: jax.Array of shape (..., H, C) and bf16/f16 dtype.
    value: jax.Array of shape (..., H, C) and bf16/f16 dtype.
    causal: bool, whether to apply causal masking.
  
  Returns:
    jax.Array of shape (..., H, C).
  """
  return jax.nn.dot_product_attention(
    query, key, value, is_causal=causal, implementation="xla")

# Pallas Kernels.
# =================

Br = 64
Bc = 64


def flash_attention_fwd_kernel(
  q_ref, 
  k_ref, 
  v_ref, 
  o_ref, 
  lse_ref, 
  *, 
  scale: float,
  num_k_blocks: int,
  causal: bool = False):
  """Forward pass kernel for flash attention.
  
  Args:
    q_ref: Slice of query tensor of shape [1, Br, C].
    k_ref: Slice of key tensor of shape [1, kv_length, C].
    v_ref: Slice of value tensor of shape [1, kv_length, C].
    o_ref: Output buffer of size [1, Br, C].
    lse_ref: Log-sum-exp buffer of size [1, Br].
    scale: Scaling factor for attention scores (usually sqrt of head dimension).

  """
  q = plgpu.load(q_ref.at[0, :, :])
  o = jnp.zeros_like(q, dtype=jnp.float32)
  m_i = jnp.full((Br,), -jnp.inf, dtype=jnp.float32)
  l_i = jnp.zeros((Br,), dtype=jnp.float32)

  q_block = pl.program_id(axis=1)
  q_pos = q_block * Br + jnp.arange(Br)

  def body(i, carry):
    o_prev, m_prev, l_prev = carry
    idx = pl.dslice(i * Bc, Bc)
    k = plgpu.load(k_ref.at[0, idx, :])
    v = plgpu.load(v_ref.at[0, idx, :])

    qk = pl.dot(q, k, trans_b=True) / scale

    if causal is not None:
      k_pos = i * Bc + jnp.arange(Bc)
      mask = q_pos[:, None] >= k_pos[None, :]
      qk = jnp.where(mask, qk, -jnp.inf)

    m_curr = jnp.max(qk, axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp(m_prev - m_next)

    s_curr = jnp.exp(qk - m_next[:, None])
    l_curr = s_curr.sum(-1)
    l_next = correction * l_prev + l_curr

    o_curr = pl.dot(s_curr.astype(v.dtype), v)
    o_next = correction[:, None] * o_prev + o_curr
    return (o_next, m_next, l_next)
  
  o, m_i, l_i = jax.lax.fori_loop(0, num_k_blocks, body, (o, m_i, l_i))
  o /= l_i[:, None]
  lse = m_i + jnp.log(l_i)

  plgpu.store(o_ref.at[0, :, :], o.astype(o_ref.dtype))
  plgpu.store(lse_ref.at[0, :], lse.astype(lse_ref.dtype))

@jax.jit
def flash_attention_fwd(query: jax.Array, key: jax.Array, value: jax.Array, causal: bool = False) -> jax.Array:
  """Flash attention forward pass using Pallas.
  
  Args:
    query: jax.Array of shape (batch..., q_length, num_heads, depth) and bf16/f16 dtype.
    key: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    value: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    causal: bool, whether to apply causal masking.
  
  Returns:
    jax.Array of shape (batch..., q_length, num_heads, depth).
  """
  bs, q_len, num_heads, head_dim = query.shape

  # Computing attention is independent across heads, so we merge the batch and head dimensions. This was the optimization in FAv2.
  bs_flat = bs * num_heads
  q_flat, k_flat, v_flat = jax.tree.map(
    lambda x: x.transpose(0, 2, 1, 3).reshape(bs_flat, q_len, head_dim), (query, key, value))
  scale = math.sqrt(head_dim)

  # Grid size eqvt to how many kernel invocations happen.
  grid = (bs_flat, pl.cdiv(q_len, Br))
  num_k_blocks = pl.cdiv(q_len, Bc)

  out_flat, lse = pl.pallas_call(
    partial(flash_attention_fwd_kernel, scale=scale, num_k_blocks=num_k_blocks, causal=causal),
    out_shape=[
      jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
      jax.ShapeDtypeStruct((bs_flat, q_len), q_flat.dtype)
    ],
    grid=grid,
    in_specs=[
      pl.BlockSpec((1, Br, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, q_len, head_dim), lambda b, _: (b, 0, 0)),
      pl.BlockSpec((1, q_len, head_dim), lambda b, _: (b, 0, 0))
    ],
    out_specs=[
      pl.BlockSpec((1, Br, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, Br), lambda b, t: (b, t))
    ],
    interpret=True,
    compiler_params=plgpu.CompilerParams(
      num_warps=4,
      num_stages=2
    )
  )(q_flat, k_flat, v_flat)

  out = out_flat.reshape(bs, num_heads, q_len, head_dim).transpose(0, 2, 1, 3)
  lse = lse.reshape(bs, num_heads, q_len)
  return out, lse