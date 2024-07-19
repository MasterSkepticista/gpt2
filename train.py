"""Trains GPT-2."""
import functools
import os

import jax
import jax.numpy as jnp
import optax
import tiktoken
from model import GPT, get_config, load_hf_pretrained
from utils import compute_flops

lead_host = jax.process_index() == 0
if os.environ.get("OMPI_COMM_WORLD_SIZE", -1) != -1:
  jax.distributed.initialize()
print("Hello from process", jax.process_index())

def info(*a, **k):
  if lead_host:
    print(*a, **k)

# @jax.jit
def sample(model, params, tokens, rng: jnp.ndarray):
  """Samples next token."""
  bs = tokens.shape[0]
  logits = model.apply({"params": params}, tokens)
  logits = logits[:, -1, :]
  topk_logits, topk_indices = jax.lax.top_k(logits, k=50)
  idx = jax.random.categorical(rng, topk_logits, axis=-1, shape=(bs, 1))
  next_token = jnp.take(topk_indices, idx)
  return next_token

def main():
  info("Total devices:", jax.device_count())
  rng = jax.random.PRNGKey(42)

  # Initialize model.
  variant = "gpt2"
  config = get_config(variant)
  model = GPT(config)
  rng, rng_init = jax.random.split(rng)

  def init(rng):
    dummy_input = jnp.ones((1, config.block_size), dtype=jnp.int32)
    params = jax.jit(model.init, backend="cpu")(rng, dummy_input)["params"]
    gflops = compute_flops(functools.partial(model.apply, {"params": params}), [dummy_input]) / 1e9
    return params, gflops

  params, gflops = init(rng_init)
  info("GFLOPs:", gflops)

  # Load GPT2 pretrained weights.
  params = load_hf_pretrained(variant)

  # Sample few tokens.
  tokenizer = tiktoken.get_encoding("gpt2")
  prompt = "Hello, I am a language model,"
  tokens = jnp.array([tokenizer.encode(prompt)])[:, :config.block_size]
  info(prompt, end="")
  while True:
    # Generate next token.
    rng, rng_sample = jax.random.split(rng)
    next_token = sample(model, params, tokens, rng_sample)
    if next_token[0] == tokenizer.eot_token:
      break
    tokens = jnp.concat([tokens, next_token], axis=-1)
    if tokens.shape[-1] > config.block_size:
      tokens = tokens[:, -config.block_size:]
    info(tokenizer.decode(next_token[0]), end="")


if __name__ == "__main__":
  main()
