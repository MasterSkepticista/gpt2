"""Dot-product Attention Kernels."""
from typing import Tuple
from functools import partial
import math

import jax
import jax.numpy as jnp

from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

@jax.jit
def dot_product_attention(
  query: jax.Array,
  key: jax.Array,
  value: jax.Array,
  mask: jax.Array,
) -> jax.Array:
  """Dot-product attention with optional masking.
  
  Args:
    query: jax.Array of shape (batch..., q_length, num_heads, depth)
    key: jax.Array of shape (batch..., kv_length, num_heads, depth)
    value: jax.Array of shape (batch..., kv_length, num_heads, depth)
    mask: jax.Array of shape (q_length, kv_length) with boolean values. True for valid positions.
  
  Returns:
    jax.Array of shape (batch..., q_length, num_heads, depth).
  """
  depth = query.shape[-1]
  attn_weights = jnp.einsum(
    "...qhd,...khd->...hqk", query, key) / jnp.sqrt(depth)
  
  if mask is not None:
    attn_weights = jnp.where(mask[None, None, :, :], attn_weights, -jnp.inf)
  
  attn_weights = jax.nn.softmax(attn_weights, axis=-1)
  out = jnp.einsum(
    "...hqk,...khd->...qhd", attn_weights, value)
  return out


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
  mask = None
  if causal:
    q_len = query.shape[-3]
    kv_len = key.shape[-3]
    assert q_len == kv_len, "For causal attention, query and key lengths must be the same"
    q_pos = jnp.arange(q_len)[:, None]
    k_pos = jnp.arange(q_len)[None, :]
    mask = q_pos >= k_pos

  return dot_product_attention(query, key, value, mask)

def cudnn_attention(
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
    query, key, value, is_causal=causal, implementation="cudnn")

# Pallas Kernels.
# =================


# Forward Pass.
# ==============

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
    q_ref: Slice of query tensor of shape [1, Br, 1, C].
    k_ref: Slice of key tensor of shape [1, kv_length, 1, C].
    v_ref: Slice of value tensor of shape [1, kv_length, 1, C].
    o_ref: Output buffer of size [1, Br, 1, C].
    lse_ref: Log-sum-exp buffer of size [1, 1, Br].
    scale: Scaling factor for attention scores (usually sqrt of head dimension).

  """
  q = plgpu.load(q_ref)
  o = jnp.zeros_like(q, dtype=jnp.float32)
  m_i = jnp.full((Br,), -jnp.inf, dtype=jnp.float32)
  l_i = jnp.zeros((Br,), dtype=jnp.float32)

  def body(i, carry):
    o_prev, m_prev, l_prev = carry
    idx = pl.dslice(i * Bc, Bc)
    k = plgpu.load(k_ref.at[idx, :])
    v = plgpu.load(v_ref.at[idx, :])

    qk = pl.dot(q, k, trans_b=True) / scale

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

  plgpu.store(o_ref, o.astype(o_ref.dtype))
  plgpu.store(lse_ref, lse.astype(lse_ref.dtype))


def flash_attention_fwd(
  query: jax.Array, 
  key: jax.Array, 
  value: jax.Array, 
  causal: bool = False
) -> Tuple[jax.Array, jax.Array]:
  """Flash attention forward pass using Pallas.
  
  Args:
    query: jax.Array of shape (batch..., q_length, num_heads, depth) and bf16/f16 dtype.
    key: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    value: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    causal: bool, whether to apply causal masking.
  
  Returns:
    Tuple (output, logsumexp).
  """
  bs, q_len, num_heads, head_dim = query.shape
  scale = math.sqrt(head_dim)

  # Grid size eqvt to how many kernel invocations happen.
  grid = (bs, num_heads, pl.cdiv(q_len, Br))
  num_k_blocks = pl.cdiv(q_len, Bc)

  out, lse = pl.pallas_call(
    partial(flash_attention_fwd_kernel, scale=scale, num_k_blocks=num_k_blocks, causal=causal),
    out_shape=[
      jax.ShapeDtypeStruct(query.shape, query.dtype),
      jax.ShapeDtypeStruct((bs, num_heads, q_len), query.dtype)
    ],
    grid=grid,
    in_specs=[
      pl.BlockSpec((None, Br, None, head_dim), lambda b, h, t: (b, t, h, 0)),
      pl.BlockSpec((None, q_len, None, head_dim), lambda b, h, _: (b, 0, h, 0)),
      pl.BlockSpec((None, q_len, None, head_dim), lambda b, h, _: (b, 0, h, 0))
    ],
    out_specs=[
      pl.BlockSpec((None, Br, None, head_dim), lambda b, h, t: (b, t, h, 0)),
      pl.BlockSpec((None, None, Br), lambda b, h, t: (b, h, t))
    ],
    interpret=True,
    compiler_params=plgpu.CompilerParams(
      num_warps=4,
      num_stages=2
    )
  )(query, key, value)

  return out, lse

# Backward Pass
# ================

def flash_attention_bwd_preprocess_kernel(o_ref, do_ref, d_ref):
  o = plgpu.load(o_ref)
  do = plgpu.load(do_ref)
  d = jnp.sum((o * do).astype(jnp.float32), axis=-1)
  plgpu.store(d_ref, d.astype(d_ref.dtype))

def flash_attention_bwd_preprocess(o_flat, do_flat):
  """Computes `D = row_sum(O * dO)`.

  Args:
    o_flat: jax.Array of shape (bs * num_heads, seqlen, head_dim)
    do_flat: jax.Array of shape (bs * num_heads, seqlen, head_dim)
  
  Returns:
    D of shape (bs * num_heads, seqlen)
  """
  bs_flat, seqlen, head_dim = o_flat.shape
  grid = (bs_flat, pl.cdiv(seqlen, Br))

  d_flat = pl.pallas_call(
    flash_attention_bwd_preprocess_kernel,
    out_shape=jax.ShapeDtypeStruct((bs_flat, seqlen), o_flat.dtype),
    grid=grid,
    in_specs=[
      pl.BlockSpec((1, Br, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, Br, head_dim), lambda b, t: (b, t, 0)),
    ],
    out_specs=pl.BlockSpec((1, Br), lambda b, t: (b, t)),
    interpret=True,
    compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2)
  )(o_flat, do_flat)
  return d_flat


def flash_attention_bwd_dkv_kernel(
  q_ref, 
  k_ref, 
  v_ref, 
  d_ref,
  do_ref,
  lse_ref,
  dk_ref,
  dv_ref,
  num_q_blocks: int,
  scale: float, 
  causal: bool):
  k = plgpu.load(k_ref.at[0, :, :])
  v = plgpu.load(v_ref.at[0, :, :])

  dk_acc = jnp.zeros(dk_ref.shape, dtype=jnp.float32)
  dv_acc = jnp.zeros(dv_ref.shape, dtype=jnp.float32)

  k_block = pl.program_id(axis=1)
  k_pos = k_block * Bc + jnp.arange(Bc)
  def body(i, carry):
    dk_acc, dv_acc = carry
    idx = pl.dslice(i * Br, Br)

    q = plgpu.load(q_ref.at[0, idx, :])
    do = plgpu.load(do_ref.at[0, idx, :])
    lse = plgpu.load(lse_ref.at[0, idx])
    d = plgpu.load(d_ref.at[0, idx])

    s = pl.dot(q, k, trans_b=True) / scale
    if causal:
      q_pos = i * Br + jnp.arange(Br)
      mask = q_pos[:, None] >= k_pos[None, :]
      s = jnp.where(mask, s, -jnp.inf)
    p = jnp.exp(s - lse[:, None])

    dp = pl.dot(do, v, trans_b=True)
    ds = p * (dp - d[:, None]) / scale

    dv_acc += pl.dot(p.astype(do.dtype), do, trans_a=True)
    dk_acc += pl.dot(ds.astype(q.dtype), q, trans_a=True)
    return dk_acc, dv_acc

  dk_acc, dv_acc = jax.lax.fori_loop(0, num_q_blocks, body, (dk_acc, dv_acc))

  plgpu.store(dk_ref, dk_acc.astype(dk_ref.dtype))
  plgpu.store(dv_ref, dv_acc.astype(dv_ref.dtype))

def flash_attention_bwd_dkv(
  q_flat: jax.Array,
  k_flat: jax.Array,
  v_flat: jax.Array,
  d_flat: jax.Array,
  do_flat: jax.Array,
  lse_flat: jax.Array,
  scale: float,
  causal: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  bs_flat, seqlen, head_dim = q_flat.shape
  num_q_blocks = pl.cdiv(seqlen, Br)
  grid = (bs_flat, pl.cdiv(seqlen, Bc))

  dk_flat, dv_flat = pl.pallas_call(
    partial(flash_attention_bwd_dkv_kernel, scale=scale, causal=causal, num_q_blocks=num_q_blocks),
    out_shape=[
      jax.ShapeDtypeStruct(k_flat.shape, k_flat.dtype),
      jax.ShapeDtypeStruct(v_flat.shape, v_flat.dtype),
    ],
    grid=grid,
    in_specs=[
      pl.BlockSpec((1, seqlen, head_dim), lambda b, t: (b, 0, 0)),
      pl.BlockSpec((1, Bc, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, Bc, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, seqlen), lambda b, t: (b, 0)),
      pl.BlockSpec((1, seqlen, head_dim), lambda b, t: (b, 0, 0)),
      pl.BlockSpec((1, seqlen), lambda b, t: (b, 0)),
    ],
    out_specs=[
      pl.BlockSpec((1, Bc, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, Bc, head_dim), lambda b, t: (b, t, 0)),
    ],
    interpret=True,
    compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2)
  )(q_flat, k_flat, v_flat, d_flat, do_flat, lse_flat)

  return dk_flat, dv_flat

def flash_attention_bwd_dq_kernel(
  q_ref,
  k_ref,
  v_ref,
  d_ref,
  do_ref,
  lse_ref,
  dq_ref,
  scale: float,
  causal: bool,
  num_kv_blocks: int,
):
  q = plgpu.load(q_ref.at[0, :, :])
  d = plgpu.load(d_ref.at[0, :])
  do = plgpu.load(do_ref.at[0, :, :])
  lse = plgpu.load(lse_ref.at[0, :])

  dq_acc = jnp.zeros(q_ref.shape, dtype=jnp.float32)

  q_block = pl.program_id(axis=1)
  q_pos = q_block * Br + jnp.arange(Br)

  def body(i, carry):
    dq_acc = carry
    idx = pl.dslice(i * Bc, Bc)
    k = plgpu.load(k_ref.at[0, idx, :])
    v = plgpu.load(v_ref.at[0, idx, :])
    
    s = pl.dot(q, k, trans_b=True) / scale

    if causal:
      k_pos = i * Bc + jnp.arange(Bc)
      mask = q_pos[:, None] >= k_pos[None, :]
      s = jnp.where(mask, s, -jnp.inf)

    p = jnp.exp(s - lse[:, None])

    dp = pl.dot(do, v, trans_b=True)
    ds = p * (dp - d[:, None]) / scale

    dq_acc += pl.dot(ds.astype(k.dtype), k)
    return dq_acc
  
  dq_acc = jax.lax.fori_loop(0, num_kv_blocks, body, dq_acc)
  plgpu.store(dq_ref, dq_acc.astype(dq_ref.dtype))

def flash_attention_bwd_dq(
  q_flat: jax.Array,
  k_flat: jax.Array,
  v_flat: jax.Array,
  d_flat: jax.Array,
  do_flat: jax.Array,
  lse_flat: jax.Array,
  scale: float,
  causal: bool = False,
) -> jax.Array:
  bs_flat, seqlen, head_dim = q_flat.shape
  num_kv_blocks = pl.cdiv(seqlen, Bc)
  grid = (bs_flat, pl.cdiv(seqlen, Br))

  dq_flat = pl.pallas_call(
    partial(flash_attention_bwd_dq_kernel, scale=scale, causal=causal, num_kv_blocks=num_kv_blocks),
    out_shape=jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
    grid=grid,
    in_specs=[
      pl.BlockSpec((1, Br, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, seqlen, head_dim), lambda b, t: (b, 0, 0)),
      pl.BlockSpec((1, seqlen, head_dim), lambda b, t: (b, 0, 0)),
      pl.BlockSpec((1, Br), lambda b, t: (b, t)),
      pl.BlockSpec((1, Br, head_dim), lambda b, t: (b, t, 0)),
      pl.BlockSpec((1, Br), lambda b, t: (b, t)),
    ],
    out_specs=pl.BlockSpec((1, Br, head_dim), lambda b, t: (b, t, 0)),
    interpret=True,
    compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
  )(q_flat, k_flat, v_flat, d_flat, do_flat, lse_flat)

  return dq_flat

def flash_attention_bwd(
  query: jax.Array,
  key: jax.Array,
  value: jax.Array,
  o: jax.Array,
  lse: jax.Array,
  do: jax.Array,
  causal: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Flash Attention Backward pass in Pallas.
  
  Args:
    query: jax.Array of shape (batch..., q_length, num_heads, depth) and bf16/f16 dtype.
    key: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    value: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    o: jax.Array of shape (batch..., q_length, num_heads, depth) - output from forward pass.
    lse: jax.Array of shape (batch..., num_heads, q_length) - logsumexp captured during forward pass.
    do: jax.Array of shape (batch..., q_length, num_heads, depth) - gradient at the output.
    causal: bool, whether to apply causal masking.

  Returns:

  """
  bs, seqlen, num_heads, head_dim = query.shape
  bs_flat = bs * num_heads
  scale = math.sqrt(head_dim)

  q_flat, k_flat, v_flat, o_flat, do_flat = jax.tree.map(
    lambda t: t.transpose(0, 2, 1, 3).reshape(bs_flat, seqlen, head_dim),
    (query, key, value, o, do)
  )
  lse_flat = lse.reshape(bs_flat, seqlen)

  # 1. Preprocess: D = row_sum(O * dO)
  d_flat = flash_attention_bwd_preprocess(o_flat, do_flat)

  # 2. Compute dK, dV
  dk_flat, dv_flat = flash_attention_bwd_dkv(
    q_flat, k_flat, v_flat, d_flat, do_flat, lse_flat, scale, causal=causal)
  
  # 3. Compute dQ
  dq_flat = flash_attention_bwd_dq(
    q_flat, k_flat, v_flat, d_flat, do_flat, lse_flat, scale, causal=causal)

  dq, dk, dv = jax.tree.map(
    lambda t: t.reshape(bs, num_heads, seqlen, head_dim).transpose(0, 2, 1, 3), 
    (dq_flat, dk_flat, dv_flat))
  return dq, dk, dv

# Register vjp
# ============
@partial(jax.custom_vjp, nondiff_argnums=(3,))
def flash_attention(
  query: jax.Array, key: jax.Array, value: jax.Array, causal: bool = False) -> jax.Array:
  """Flash attention using Pallas.
  
  Args:
    query: jax.Array of shape (batch..., q_length, num_heads, depth) and bf16/f16 dtype.
    key: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    value: jax.Array of shape (batch..., kv_length, num_heads, depth) and bf16/f16 dtype.
    causal: bool, whether to apply causal masking.
  
  Returns:
    jax.Array of shape (batch..., q_length, num_heads, depth).
  """
  o, _ = flash_attention_fwd(query, key, value, causal=causal)
  return o

def flash_attention_fwd_rule(query, key, value, causal):
  o, lse = flash_attention_fwd(query, key, value, causal=causal)
  return o, (query, key, value, o, lse)

def flash_attention_bwd_rule(causal, res, g):
  query, key, value, o, lse = res
  dq, dk, dv = flash_attention_bwd(query, key, value, o, lse, g, causal=causal)
  return dq, dk, dv

flash_attention.defvjp(flash_attention_fwd_rule, flash_attention_bwd_rule)