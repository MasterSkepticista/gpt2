"""Dot-product Attention Kernels."""
import os
from dataclasses import dataclass
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

INTERPRET_MODE = os.environ.get("INTERPRET_MODE", "0") == "1"

@dataclass(frozen=True)
class KernelConfig:
  block_q: int = 64
  block_kv: int = 64
  num_warps: int = 4
  num_stages: int = 2


DEFAULT_FWD_CONFIG = KernelConfig(block_q=128, block_kv=64, num_warps=4, num_stages=3)
DEFAULT_BWD_CONFIG = KernelConfig(block_q=64, block_kv=64, num_warps=4, num_stages=2)


def flash_attention_fwd_kernel(
  q_ref, 
  k_ref, 
  v_ref, 
  o_ref, 
  lse_ref, 
  *, 
  scale: float,
  block_q: int,
  block_kv: int,
  causal: bool = False
):
  del causal
  seq_len = k_ref.shape[0]
  q = plgpu.load(q_ref)
  o = jnp.zeros_like(q, dtype=jnp.float32)
  m_i = jnp.zeros((block_q,), dtype=jnp.float32) - float("inf")
  l_i = jnp.zeros((block_q,), dtype=jnp.float32)

  def body(i, carry):
    o_prev, m_prev, l_prev = carry
    idx = pl.dslice(i * block_kv, block_kv)
    k = plgpu.load(k_ref.at[idx, :])
    v = plgpu.load(v_ref.at[idx, :])

    qk_scale = math.log2(math.e) / scale
    qk = pl.dot(q, k.T) * qk_scale

    m_curr = jnp.max(qk, axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp2(m_prev - m_next)

    s_curr = jnp.exp2(qk - m_next[:, None])
    l_curr = s_curr.sum(-1)
    l_next = correction * l_prev + l_curr

    o_curr = pl.dot(s_curr.astype(v.dtype), v)
    o_next = correction[:, None] * o_prev + o_curr
    return (o_next, m_next, l_next)
  
  num_k_blocks = pl.cdiv(seq_len, block_kv)
  o, m_i, l_i = jax.lax.fori_loop(0, num_k_blocks, body, (o, m_i, l_i))
  o /= l_i[:, None]
  lse = m_i + jnp.log2(l_i)

  plgpu.store(o_ref, o.astype(o_ref.dtype))
  plgpu.store(lse_ref, lse.astype(lse_ref.dtype))


def flash_attention_fwd(
  query: jax.Array, 
  key: jax.Array, 
  value: jax.Array, 
  causal: bool = False,
  config: KernelConfig = DEFAULT_FWD_CONFIG,
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
  block_q = config.block_q
  block_kv = config.block_kv

  grid = (pl.cdiv(q_len, block_q), bs, num_heads)

  out, lse = pl.pallas_call(
    partial(
      flash_attention_fwd_kernel,
      scale=scale,
      block_q=block_q,
      block_kv=block_kv,
      causal=causal,
    ),
    out_shape=[
      jax.ShapeDtypeStruct(query.shape, query.dtype),
      jax.ShapeDtypeStruct((bs, num_heads, q_len), query.dtype)
    ],
    grid=grid,
    in_specs=[
      pl.BlockSpec((None, block_q, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, q_len, None, head_dim), lambda t, b, h: (b, 0, h, 0)),
      pl.BlockSpec((None, q_len, None, head_dim), lambda t, b, h: (b, 0, h, 0))
    ],
    out_specs=[
      pl.BlockSpec((None, block_q, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, None, block_q), lambda t, b, h: (b, h, t))
    ],
    interpret=INTERPRET_MODE,
    compiler_params=plgpu.CompilerParams(
      num_warps=config.num_warps,
      num_stages=config.num_stages
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

def flash_attention_bwd_preprocess(o, do, config: KernelConfig = DEFAULT_BWD_CONFIG):
  """Computes `D = row_sum(O * dO)`.

  Args:
    o: jax.Array of shape (bs, seqlen, num_heads, head_dim)
    do: jax.Array of shape (bs, seqlen, num_heads, head_dim)
  
  Returns:
    D of shape (bs, seqlen, num_heads)
  """
  bs, seqlen, num_heads, head_dim = o.shape
  block_q = config.block_q
  grid = (pl.cdiv(seqlen, block_q), bs, num_heads)

  return pl.pallas_call(
    flash_attention_bwd_preprocess_kernel,
    out_shape=jax.ShapeDtypeStruct((bs, seqlen, num_heads), o.dtype),
    grid=grid,
    in_specs=[
      pl.BlockSpec((None, block_q, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, block_q, None, head_dim), lambda t, b, h: (b, t, h, 0)),
    ],
    out_specs=pl.BlockSpec((None, block_q, None), lambda t, b, h: (b, t, h)),
    interpret=INTERPRET_MODE,
    compiler_params=plgpu.CompilerParams(
      num_warps=config.num_warps,
      num_stages=config.num_stages,
    )
  )(o, do)


def flash_attention_bwd_dkv_kernel(
  q_ref, 
  k_ref, 
  v_ref, 
  d_ref,
  do_ref,
  lse_ref,
  dk_ref,
  dv_ref,
  *,
  block_q: int,
  scale: float, 
  causal: bool
):
  del causal
  seq_len = q_ref.shape[0]
  k = plgpu.load(k_ref)
  v = plgpu.load(v_ref)

  dk_acc = jnp.zeros(dk_ref.shape, dtype=jnp.float32)
  dv_acc = jnp.zeros(dv_ref.shape, dtype=jnp.float32)

  def body(i, carry):
    dk_acc, dv_acc = carry
    idx = pl.dslice(i * block_q, block_q)

    q = plgpu.load(q_ref.at[idx, :])
    do = plgpu.load(do_ref.at[idx, :])
    lse = plgpu.load(lse_ref.at[idx])
    d = plgpu.load(d_ref.at[idx])

    qk_scale = math.log2(math.e) / scale
    s = pl.dot(q, k.T) * qk_scale
    p = jnp.exp2(s - lse[:, None])

    dp = pl.dot(do, v.T)
    ds = p * (dp - d[:, None]) / scale

    dv_acc += pl.dot(p.T.astype(do.dtype), do)
    dk_acc += pl.dot(ds.T.astype(q.dtype), q)
    return dk_acc, dv_acc

  num_q_blocks = pl.cdiv(seq_len, block_q)
  dk_acc, dv_acc = jax.lax.fori_loop(0, num_q_blocks, body, (dk_acc, dv_acc))

  plgpu.store(dk_ref, dk_acc.astype(dk_ref.dtype))
  plgpu.store(dv_ref, dv_acc.astype(dv_ref.dtype))

def flash_attention_bwd_dkv(
  query: jax.Array,
  key: jax.Array,
  value: jax.Array,
  d: jax.Array,
  do: jax.Array,
  lse: jax.Array,
  scale: float,
  causal: bool = False,
  config: KernelConfig = DEFAULT_BWD_CONFIG,
) -> Tuple[jax.Array, jax.Array]:
  bs, seqlen, num_heads, head_dim = query.shape
  block_q = config.block_q
  block_kv = config.block_kv
  grid = (pl.cdiv(seqlen, block_kv), bs, num_heads)

  return pl.pallas_call(
    partial(
      flash_attention_bwd_dkv_kernel,
      scale=scale,
      block_q=block_q,
      causal=causal,
    ),
    out_shape=[
      jax.ShapeDtypeStruct(key.shape, key.dtype),
      jax.ShapeDtypeStruct(value.shape, value.dtype),
    ],
    grid=grid,
    in_specs=[
      pl.BlockSpec((None, seqlen, None, head_dim), lambda t, b, h: (b, 0, h, 0)),
      pl.BlockSpec((None, block_kv, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, block_kv, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, seqlen, None), lambda t, b, h: (b, 0, h)),
      pl.BlockSpec((None, seqlen, None, head_dim), lambda t, b, h: (b, 0, h, 0)),
      pl.BlockSpec((None, None, seqlen), lambda t, b, h: (b, h, 0)),
    ],
    out_specs=[
      pl.BlockSpec((None, block_kv, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, block_kv, None, head_dim), lambda t, b, h: (b, t, h, 0)),
    ],
    interpret=INTERPRET_MODE,
    compiler_params=plgpu.CompilerParams(
      num_warps=config.num_warps,
      num_stages=config.num_stages,
    )
  )(query, key, value, d, do, lse)

def flash_attention_bwd_dq_kernel(
  q_ref,
  k_ref,
  v_ref,
  d_ref,
  do_ref,
  lse_ref,
  dq_ref,
  *,
  scale: float,
  block_kv: int,
  causal: bool,
):
  del causal
  seq_len = k_ref.shape[0]
  q = plgpu.load(q_ref)
  d = plgpu.load(d_ref)
  do = plgpu.load(do_ref)
  lse = plgpu.load(lse_ref)

  dq_acc = jnp.zeros(q_ref.shape, dtype=jnp.float32)

  def body(i, carry):
    dq_acc = carry
    idx = pl.dslice(i * block_kv, block_kv)
    k = plgpu.load(k_ref.at[idx, :])
    v = plgpu.load(v_ref.at[idx, :])
    
    qk_scale = math.log2(math.e) / scale
    s = pl.dot(q, k.T) * qk_scale
    p = jnp.exp2(s - lse[:, None])

    dp = pl.dot(do, v.T)
    ds = p * (dp - d[:, None]) / scale

    dq_acc += pl.dot(ds.astype(k.dtype), k)
    return dq_acc
  
  num_kv_blocks = pl.cdiv(seq_len, block_kv)
  dq_acc = jax.lax.fori_loop(0, num_kv_blocks, body, dq_acc)
  plgpu.store(dq_ref, dq_acc.astype(dq_ref.dtype))

def flash_attention_bwd_dq(
  query: jax.Array,
  key: jax.Array,
  value: jax.Array,
  d: jax.Array,
  do: jax.Array,
  lse: jax.Array,
  scale: float,
  causal: bool = False,
  config: KernelConfig = DEFAULT_BWD_CONFIG,
) -> jax.Array:
  bs, seqlen, num_heads, head_dim = query.shape
  block_q = config.block_q
  block_kv = config.block_kv
  grid = (pl.cdiv(seqlen, block_q), bs, num_heads)

  return pl.pallas_call(
    partial(
      flash_attention_bwd_dq_kernel,
      scale=scale,
      causal=causal,
      block_kv=block_kv,
    ),
    out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
    grid=grid,
    in_specs=[
      pl.BlockSpec((None, block_q, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, seqlen, None, head_dim), lambda t, b, h: (b, 0, h, 0)),
      pl.BlockSpec((None, seqlen, None, head_dim), lambda t, b, h: (b, 0, h, 0)),
      pl.BlockSpec((None, block_q, None), lambda t, b, h: (b, t, h)),
      pl.BlockSpec((None, block_q, None, head_dim), lambda t, b, h: (b, t, h, 0)),
      pl.BlockSpec((None, None, block_q), lambda t, b, h: (b, h, t)),
    ],
    out_specs=pl.BlockSpec((None, block_q, None, head_dim), lambda t, b, h: (b, t, h, 0)),
    interpret=INTERPRET_MODE,
    compiler_params=plgpu.CompilerParams(
      num_warps=config.num_warps,
      num_stages=config.num_stages,
    ),
  )(query, key, value, d, do, lse)

def flash_attention_bwd(
  query: jax.Array,
  key: jax.Array,
  value: jax.Array,
  o: jax.Array,
  lse: jax.Array,
  do: jax.Array,
  causal: bool = False,
  config: KernelConfig = DEFAULT_BWD_CONFIG,
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
  head_dim = query.shape[-1]
  scale = math.sqrt(head_dim)

  # 1. Preprocess: D = row_sum(O * dO)
  D = flash_attention_bwd_preprocess(o, do, config=config)

  # 2. Compute dK, dV
  dk, dv = flash_attention_bwd_dkv(
    query, key, value, D, do, lse, scale, causal=causal, config=config)
  
  # 3. Compute dQ
  dq = flash_attention_bwd_dq(
    query, key, value, D, do, lse, scale, causal=causal, config=config)

  return dq, dk, dv


def make_flash_attention(
  *,
  fwd_config: KernelConfig = DEFAULT_FWD_CONFIG,
  bwd_config: KernelConfig = DEFAULT_BWD_CONFIG,
):
  @partial(jax.custom_vjp, nondiff_argnums=(3,))
  def flash_attention_impl(
    query: jax.Array, key: jax.Array, value: jax.Array, causal: bool = False
  ) -> jax.Array:
    o, _ = flash_attention_fwd(query, key, value, causal=causal, config=fwd_config)
    return o

  def flash_attention_fwd_rule(query, key, value, causal):
    o, lse = flash_attention_fwd(query, key, value, causal=causal, config=fwd_config)
    return o, (query, key, value, o, lse)

  def flash_attention_bwd_rule(causal, res, g):
    query, key, value, o, lse = res
    dq, dk, dv = flash_attention_bwd(
      query, key, value, o, lse, g, causal=causal, config=bwd_config
    )
    return dq, dk, dv

  flash_attention_impl.defvjp(flash_attention_fwd_rule, flash_attention_bwd_rule)
  return flash_attention_impl

# Register vjp
# ============
flash_attention = make_flash_attention()
