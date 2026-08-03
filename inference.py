"""Generate text from a Hugging Face GPT-2 model.

Additionally requires HF-transformers package to pull weights:
`pip install torch transformers`.

Usage:
```bash
# Weights will be cached in the workdir, reused for inference.
python inference.py --workdir /tmp/gpt2
```
"""
import os

from absl import app, flags
from flax import serialization
import jax
import jax.numpy as jnp
import tiktoken

from model import GPT, load_hf_pretrained


PROMPT = "The IEEE-754 standard specifies"
MAX_NEW_TOKENS = 100
TOP_K = 50
RNG_SEED = 42
PARAMS_FILENAME = "hf_gpt2_params.msgpack"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "workdir",
    default=None,
    help="Directory used to cache pretrained model parameters.",
)

flags.DEFINE_string(
    "prompt",
    default=PROMPT,
    help="Prompt to generate text from.",
)


def load_params():
  """Restores cached parameters or downloads and optionally caches them."""
  if FLAGS.workdir:
    params_path = os.path.join(FLAGS.workdir, PARAMS_FILENAME)
    if os.path.exists(params_path):
      with open(params_path, "rb") as f:
        print(f"Loading parameters from {params_path}")
        return serialization.msgpack_restore(f.read())

  params = load_hf_pretrained("gpt2")
  if FLAGS.workdir:
    os.makedirs(FLAGS.workdir, exist_ok=True)
    with open(params_path, "wb") as f:
      f.write(serialization.to_bytes(params))
    print(f"Saved parameters to {params_path}")
  return params


def main(unused_argv):
  del unused_argv

  # Initialize model.
  model = GPT(
      # GPT-2 family config
      vocab_size=50_257,
      block_size=1024,
      # GPT-2 124M model-specific config
      emb_dim=768,
      num_heads=12,
      num_layers=12,
      sdpa_implementation=None,
      # dtype of computation.
      dtype=jnp.float32,
  )
  rng = jax.random.PRNGKey(RNG_SEED)
  params = load_params()

  @jax.jit
  def sample_next(params, tokens, token_index, sample_rng):
    logits = model.apply({"params": params}, tokens)
    logits = logits[jnp.arange(tokens.shape[0]), token_index]
    topk_logits, topk_indices = jax.lax.top_k(logits, k=TOP_K)
    sampled_index = jax.random.categorical(sample_rng, topk_logits, axis=-1)
    return topk_indices[jnp.arange(tokens.shape[0]), sampled_index]

  encoder = tiktoken.get_encoding("gpt2")
  token_ids = encoder.encode(FLAGS.prompt)
  if not token_ids:
    raise ValueError("PROMPT must encode to at least one token.")
  token_ids = token_ids[-model.block_size:]

  print("--------------")
  print(FLAGS.prompt, end="", flush=True)
  tokens = jnp.zeros((1, model.block_size), dtype=jnp.int32)
  tokens = tokens.at[0, :len(token_ids)].set(jnp.asarray(token_ids, dtype=jnp.int32))
  for token_index in range(len(token_ids) - 1, min(len(token_ids) + MAX_NEW_TOKENS - 1,
                                                      model.block_size - 1)):
    rng, sample_rng = jax.random.split(rng)
    next_token = int(sample_next(params, tokens, token_index, sample_rng)[0])
    tokens = tokens.at[0, token_index + 1].set(next_token)
    print(encoder.decode([next_token]), end="", flush=True)
  print()


if __name__ == "__main__":
  app.run(main)
