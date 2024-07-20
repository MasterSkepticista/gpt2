"""Trains GPT-2."""
import functools
import os

import jax
from flax import jax_utils
import jax.numpy as jnp
import optax
import tensorflow as tf
import tiktoken
from model import GPT, get_config, load_hf_pretrained
import utils as u

if os.environ.get("OMPI_COMM_WORLD_SIZE", -1) != -1:
  jax.distributed.initialize()
lead_host = jax.process_index() == 0
print("Hello from process", jax.process_index())


def info(*a, **k):
  if lead_host:
    print(*a, **k)


def build_pipeline(data_dir: str,
                   batch_size: int,
                   block_size: int,
                   train: bool = False):
  """Builds tf.data pipeline."""
  
  split = "train" if train else "val"
  ds = tf.data.Dataset.list_files(os.path.join(data_dir, f"{split}_*.tfrecord"))
  
  ds = ds.shard(jax.device_count(), jax.process_index())
  ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.repeat()
  ds = ds.shuffle(10_000)

  def _decode(proto):
    ex = tf.io.parse_single_example(proto, {"tokens": tf.io.FixedLenFeature([], tf.string)})
    return tf.io.decode_raw(ex["tokens"], tf.uint16)

  ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.unbatch()  # Flattens to a stream of tokens.
  ds = ds.batch(block_size + 1, drop_remainder=True)
  ds = ds.map(lambda x: (x[:-1], x[1:]))  # Input and Next (completion) pairs.
  ds = ds.batch(batch_size, drop_remainder=True)

  ds = ds.prefetch(tf.data.AUTOTUNE)

  # Shard batches to GPUs.
  ds = iter(ds)
  ds = map(u.tf_to_numpy, ds)
  ds = map(u.shard_batches, ds)
  ds = jax_utils.prefetch_to_device(ds, 2)
  return ds

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
  cfg = get_config(variant)
  model = GPT(cfg)
  rng, rng_init = jax.random.split(rng)

  def init(rng):
    dummy_input = jnp.ones((1, cfg.block_size), dtype=jnp.int32)
    params = jax.jit(model.init, backend="cpu")(rng, dummy_input)["params"]
    gflops = u.compute_flops(
        functools.partial(model.apply, {"params": params}), [dummy_input]) / 1e9
    return params, gflops

  params, gflops = init(rng_init)
  info("GFLOPs:", gflops)

  # Load GPT2 pretrained weights.
  params = load_hf_pretrained(variant)

  # Sample few tokens.
  if False:
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

  # Build data pipeline.
  train_iter = build_pipeline("data", batch_size=32, block_size=cfg.block_size, train=True)
  val_iter = build_pipeline("data", batch_size=32, block_size=cfg.block_size, train=False)

  # Build optimizer and train state.
  max_lr = 6e-4
  min_lr = 0.1 * max_lr
  max_steps = 19073
  warmup_steps = 715
  sched_fn = u.get_cosine_lr_schedule(max_lr, min_lr, max_steps, warmup_steps)

  tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
      learning_rate=sched_fn, 
      b1=0.9, 
      b2=0.95,
      weight_decay=0.1,
      mask=jax.tree.map(lambda p: p.ndim > 1, params)),
  )
  opt_state = tx.init(params)


if __name__ == "__main__":
  main()
