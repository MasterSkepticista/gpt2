"""Trains GPT-2."""
import functools
import os
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tiktoken
import utils as u
from absl import app, flags, logging
from clu import metric_writers, periodic_actions
from flax import jax_utils, struct
from model import GPT, get_config, load_hf_pretrained

logging.set_verbosity("info")

flags.DEFINE_string("workdir", 
                    default=None, 
                    required=True, 
                    help="Path to store logs/checkpoints.")
flags.mark_flags_as_required(["workdir"])
FLAGS = flags.FLAGS

PyTree = Any

@struct.dataclass
class TrainState:
  """Simple container to hold training state."""
  params: PyTree
  opt_state: PyTree
  global_step: int
  tx: optax.GradientTransformation = struct.field(pytree_node=False)

def build_pipeline(data_dir: str,
                   batch_size: int,
                   block_size: int,
                   train: bool = False):
  """Builds tf.data pipeline."""

  split = "train" if train else "val"
  ds = tf.data.Dataset.list_files(os.path.join(data_dir, f"{split}_*.tfrecord"))

  if train:
    ds = ds.shuffle(256)  # At shards-level.

  ds = tf.data.TFRecordDataset(ds)
  ds = ds.shard(jax.process_count(), jax.process_index())
  ds = ds.repeat()
  
  if train:
    ds = ds.shuffle(10_000)  # At documents-level.

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

def get_train_step(model: Any, grad_accum_steps: int):
  """Returns a fn that executes one step of training."""

  def train_step(state: TrainState, batch: PyTree) -> Tuple[TrainState, PyTree]:
    """Single update step."""
    measurements = {}

    def loss_fn(params, batch):
      x, y = batch
      logits = model.apply({"params": params}, x)
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
      return loss.mean()

    # Compute local gradient.
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = u.accumulate_gradient(grad_fn, state.params, batch, grad_accum_steps)
    grads = jax.lax.pmean(grads, axis_name="batch")

    # Compute and apply updates.
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Update state.
    new_state = state.replace(
      params=new_params,
      opt_state=new_opt_state,
      global_step=state.global_step + 1)

    # Metrics.
    measurements["loss"] = loss
    gs = jax.tree.leaves(grads)
    measurements["l2_grads"] = jnp.sqrt(sum(jnp.vdot(g, g) for g in gs))
    us = jax.tree.leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum(jnp.vdot(u, u) for u in us))
    ps = jax.tree.leaves(new_params)
    measurements["l2_params"] = jnp.sqrt(sum(jnp.vdot(p, p) for p in ps))
    return new_state, measurements

  return train_step

def get_eval_step(model: Any):
  """Returns a fn that runs one step of eval."""

  def eval_step(state: TrainState, batch: PyTree) -> dict:
    x, y = batch
    logits = model.apply({"params": state.params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return {"loss": loss.mean()}
  
  return eval_step

def main(unused_argv):
  if os.environ.get("OMPI_COMM_WORLD_SIZE", -1) != -1:
    jax.distributed.initialize()
  lead_host = jax.process_index() == 0
  logging.info("Hello from process %d holding %d device(s)", jax.process_index(), jax.local_device_count())

  def info(s, *a):
    if lead_host:
      logging.info("\u001b[32mNOTE\u001b[0m: " + s, *a)

  info("Total devices: %d", jax.device_count())
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
  info(f"GFLOPs for model {variant}: {gflops:.4f}")

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
  tokens_per_batch = int(2**17)  # 0.5M
  assert tokens_per_batch % cfg.block_size == 0
  batch_size = tokens_per_batch // cfg.block_size
  info("Number of tokens per batch (globally): %d", tokens_per_batch)
  info("Per-device batch size: %d", batch_size)
  train_iter = build_pipeline("data", batch_size, cfg.block_size, train=True)
  val_iter = build_pipeline("data", batch_size, cfg.block_size, train=False)

  # Build optimizer and train state.
  max_lr = 6e-4
  min_lr = 0.1 * max_lr
  max_steps = 19073
  warmup_steps = 715
  sched_fn = u.get_cosine_lr_schedule(max_lr, min_lr, max_steps, warmup_steps)

  tx = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          sched_fn,
          b1=0.9,
          b2=0.95,
          weight_decay=0.1,
          mask=jax.tree.map(lambda p: p.ndim > 1, params)))
  opt_state = tx.init(params)

  # Resume from checkpoint or start from scratch.
  start_step = 0
  state = TrainState(params, opt_state, start_step, tx)
  state = jax_utils.replicate(state)
  train_step = jax.pmap(get_train_step(model), axis_name="batch")
  eval_step = jax.pmap(get_eval_step(model), axis_name="batch")
  
  # Logging setup.
  log_summary_steps = 50
  log_eval_steps = 500
  writer = metric_writers.AsyncWriter(metric_writers.SummaryWriter(FLAGS.workdir))
  progress = periodic_actions.ReportProgress(
    num_train_steps=max_steps, writer=writer, every_steps=log_summary_steps)
  hooks = []
  if lead_host:
    hooks.append(progress)

  # Train loop.
  train_metrics = []
  info("Starting training at step %d", start_step + 1)
  for step in range(start_step + 1, max_steps + 1):

    # Train step.
    train_batch = next(train_iter)
    state, metrics = train_step(state, train_batch)
    train_metrics.append(jax_utils.unreplicate(metrics))

    for h in hooks:
      h(step)

    # Log train stats.
    if step % log_summary_steps == 0:
      extra_logs = {"global_schedule": sched_fn(step)}
      u.log_summary(step, train_metrics, extra_logs=extra_logs, writer=writer, prefix="train")
      train_metrics = []
    
    # Evaluate and store checkpoints.
    if step % log_eval_steps == 0:
      info("Running eval")
      with progress.timed("eval"):
        eval_metrics = []
        for _ in range(20):
          eval_batch = next(val_iter)
          eval_metrics.append(jax_utils.unreplicate(eval_step(state, eval_batch)))
        u.log_summary(step, eval_metrics, writer=writer, prefix="val")

if __name__ == "__main__":
  app.run(main)
