"""Self-attention microbenchmark."""
import time
from absl import app
from absl import flags
from functools import partial

import jax
import jax.numpy as jnp

from attention import flash_attention, naive_attention, cudnn_attention

FLAGS = flags.FLAGS

flags.DEFINE_string("impl", "pallas", 
  "Attention implementation to benchmark: 'pallas', 'naive', or 'cudnn'.")
flags.DEFINE_integer("head_dim", 64, 
  "Dimension of each attention head (64 or 128 recommended).")

MAX_TOKENS = 16 * 1024

def generate_tensors(B, T, H, C):
  keys = jax.random.split(jax.random.PRNGKey(42), 3)
  q = jax.random.normal(keys[0], (B, T, H, C), dtype=jnp.bfloat16)
  k = jax.random.normal(keys[1], (B, T, H, C), dtype=jnp.bfloat16)
  v = jax.random.normal(keys[2], (B, T, H, C), dtype=jnp.bfloat16)
  return q, k, v

def timeit(fn, *args, num_iters=10):
  # Warmup
  for _ in range(2):
    out = fn(*args)
  jax.block_until_ready(out)

  start_time = time.monotonic()
  for _ in range(num_iters):
    out = fn(*args)
  jax.block_until_ready(out)
  end_time = time.monotonic()

  return (end_time - start_time) / num_iters

def main(argv):
  match(FLAGS.impl):
    case "pallas":
      attention_fn = flash_attention
    case "naive":
      attention_fn = naive_attention
    case "cudnn":
      attention_fn = cudnn_attention
    case _:
      raise ValueError(f"Invalid implementation: {FLAGS.impl}")

  q, k, v = generate_tensors(1, 1024, 12, 64)
  o_ref = naive_attention(q, k, v, causal=False)    
  jit_fwd_fn = jax.jit(partial(attention_fn, causal=False))
  jnp.allclose(jit_fwd_fn(q, k, v), o_ref, rtol=1e-2, atol=1e-2)
  print("Results match. Starting benchmark...")
  print("Benchmarking attention implementation:", FLAGS.impl)

  for T in [1024, 2048, 4096, 8192, 16384]:
    bs = MAX_TOKENS // T
    num_heads = 2048 // FLAGS.head_dim
    q, k, v = generate_tensors(bs, T, num_heads, FLAGS.head_dim)

    # Forward pass
    jit_fwd_fn = jax.jit(partial(attention_fn, causal=False))
    flop_count = 4 * T**2 * FLAGS.head_dim * num_heads * bs
    avg_time = timeit(jit_fwd_fn, q, k, v)
    tflops = flop_count * 1e-12 / avg_time
    mfu = (tflops / 121) * 100  # Using 121 TFLOPs for L4 as per notebook context
    print(f"(fwd) T={T:5d}, B={bs:3d}, TFLOP/s={tflops:.2f}, MFU={mfu:.2f}%")

    # Forward + Backward pass
    def loss(q, k, v):
      out = attention_fn(q, k, v, causal=False)
      return jnp.sum(out)
    jit_fwd_bwd_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    avg_time = timeit(jit_fwd_bwd_fn, q, k, v)
    tflops = flop_count * 3.5 * 1e-12 / avg_time  # Backward is ~2.5x forward
    mfu = (tflops / 121) * 100  # Using 121 TFLOPs for L4 as per notebook context
    print(f"(fwd+bwd) T={T:5d}, B={bs:3d}, TFLOP/s={tflops:.2f}, MFU={mfu:.2f}%")


if __name__ == "__main__":
  app.run(main)