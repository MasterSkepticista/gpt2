"""Trains GPT-2."""
import os
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import utils as u
from absl import app, flags, logging
from clu import metric_writers, periodic_actions
from clu.parameter_overview import get_parameter_overview
from flax import jax_utils
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from flax.training.train_state import TrainState
from ml_collections import config_flags
from model import GPT, load_hf_pretrained

logging.set_verbosity("info")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
flax.config.update("flax_use_orbax_checkpointing", False)

flags.DEFINE_string("workdir", 
                    default=None,
                    help="Path to store logs/checkpoints.")

config_flags.DEFINE_config_file("config", 
                                default=None, 
                                help_string="Training config.", 
                                lock_config=True)

flags.mark_flags_as_required(["workdir", "config"])

FLAGS = flags.FLAGS


def build_pipeline(data_dir: str,
                   batch_size: int,
                   block_size: int,
                   shuffle_seed: int,
                   train: bool = False):
  """Builds tf.data pipeline."""

  split = "train" if train else "val"
  ds = tf.data.Dataset.list_files(
    os.path.join(data_dir, f"{split}_*.tfrecord"), shuffle=False)

  if train:
    ds = ds.shuffle(256, seed=shuffle_seed)  # At shards-level.

  ds = tf.data.TFRecordDataset(ds)
  ds = ds.shard(jax.process_count(), jax.process_index())
  ds = ds.repeat()
  
  if train:
    ds = ds.shuffle(10_000, seed=shuffle_seed)  # At documents-level.

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
def sample(state, tokens, rng: jnp.ndarray):
  """Samples next token."""
  bs = tokens.shape[0]
  logits = state.apply_fn({"params": state.params}, tokens)
  logits = logits[:, -1, :]
  topk_logits, topk_indices = jax.lax.top_k(logits, k=50)
  idx = jax.random.categorical(rng, topk_logits, axis=-1, shape=(bs, 1))
  next_token = jnp.take(topk_indices, idx)
  return next_token

def get_train_step(grad_accum_steps: int):
  """Returns a fn that executes one step of training."""

  def train_step(state: TrainState, batch: Any) -> Tuple[TrainState, dict]:
    """Single update step."""
    measurements = {}

    def loss_fn(params, batch):
      x, y = batch
      logits = state.apply_fn({"params": params}, x)
      # FIXME: Quantization bug in `optax.softmax_ce_with_integer_labels` 
      # if logits are `bfloat16`.
      # Relevant issue: https://github.com/google-deepmind/optax/issues/1020
      # logits = jnp.asarray(logits, jnp.float32)
      y_oh = jax.nn.one_hot(y, logits.shape[-1])
      loss = optax.softmax_cross_entropy(logits, y_oh)
      return loss.mean()

    # Compute gradients and apply updates.
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = u.accumulate_gradient(grad_fn, state.params, batch, grad_accum_steps)
    loss, grads = jax.lax.pmean((loss, grads), axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    # Metrics.
    measurements["loss"] = loss
    gs = jax.tree.leaves(grads)
    measurements["l2_grads"] = jnp.sqrt(sum(jnp.vdot(g, g) for g in gs))
    ps = jax.tree.leaves(new_state.params)
    measurements["l2_params"] = jnp.sqrt(sum(jnp.vdot(p, p) for p in ps))
    return new_state, measurements

  return train_step

def get_eval_step():
  """Returns a fn that runs one step of eval."""

  def eval_step(state: TrainState, batch: Any) -> dict:
    x, y = batch
    logits = state.apply_fn({"params": state.params}, x)
    # FIXME: Quantization bug in `optax.softmax_ce_with_integer_labels` 
    # if logits are `bfloat16`.
    # Relevant issue: https://github.com/google-deepmind/optax/issues/1020
    y_oh = jax.nn.one_hot(y, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, y_oh)
    return {"loss": loss.mean()}
  
  return eval_step

def main(unused_argv):
  if os.environ.get("OMPI_COMM_WORLD_SIZE", -1) != -1:
    jax.distributed.initialize()
  lead_host = jax.process_index() == 0
  logging.info("Hello from process %d holding %d device(s)", 
                jax.process_index(), jax.local_device_count())

  def info(s, *a):
    if lead_host:
      logging.info("\u001b[32mNOTE\u001b[0m: " + s, *a)

  info("Total devices: %d", jax.device_count())

  # Initialize model.
  cfg = FLAGS.config
  model = GPT(**cfg.model)
  rng = jax.random.PRNGKey(cfg.rng_seed)
  rng, rng_init = jax.random.split(rng)

  def init(rng):
    dummy_input = jnp.ones((1, cfg.model.block_size), dtype=jnp.int32)
    params = jax.jit(model.init)(rng, dummy_input)["params"]
    gflops = u.compute_flops(model.apply, [{"params": params}, dummy_input]) / 1e9
    return params, gflops

  params, gflops = init(rng_init)
  info(get_parameter_overview(params))
  info(f"GFLOPs for model: {gflops:.4f}")

  # Build data pipeline.
  local_batch_size = cfg.batch_size // jax.process_count()
  info("Batch size: %d", cfg.batch_size)
  info("Tokens per batch: %d", cfg.batch_size * cfg.model.block_size)
  train_iter = build_pipeline(
    cfg.data_dir, local_batch_size, cfg.model.block_size, cfg.rng_seed, train=True)
  val_iter = build_pipeline(
    cfg.data_dir, local_batch_size, cfg.model.block_size, cfg.rng_seed, train=False)

  # Build optimizer and train state.
  sched_fn = u.get_cosine_lr_schedule(
    cfg.lr, cfg.min_lr, cfg.total_steps, cfg.warmup_steps)

  tx = optax.chain(
    optax.clip_by_global_norm(cfg.grad_clip_norm),
    optax.adamw(
      sched_fn, 
      **cfg.optax_kwargs, 
      mask=jax.tree.map(lambda p: p.ndim > 1, params)))

  # Resume from checkpoint or start from scratch.
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  state = restore_checkpoint(FLAGS.workdir, state)
  start_step = state.step

  # Replicate.
  state = jax_utils.replicate(state)
  train_step = jax.pmap(get_train_step(cfg.grad_accum_steps), axis_name="batch")
  eval_step = jax.pmap(get_eval_step(), axis_name="batch")
  
  # Logging setup.
  writer = metric_writers.AsyncWriter(metric_writers.SummaryWriter(FLAGS.workdir))
  progress = periodic_actions.ReportProgress(
    num_train_steps=cfg.total_steps, 
    writer=writer, 
    every_steps=cfg.log_train_steps, 
    every_secs=None)
  profile = periodic_actions.Profile(
    logdir=FLAGS.workdir, num_profile_steps=5, every_secs=None)
  hooks = []
  if lead_host:
    hooks.extend((progress, profile))

  # Train loop.
  train_metrics = []
  info("Starting training loop at step %d", start_step + 1)
  for step in range(start_step + 1, cfg.total_steps + 1):

    # Train step.
    train_batch = next(train_iter)
    state, metrics = train_step(state, train_batch)
    train_metrics.append(u.unreplicate_and_get(metrics))

    for h in hooks:
      h(step)

    # Log train stats.
    if step % cfg.log_train_steps == 0:
      extra_logs = {"global_schedule": sched_fn(step)}
      u.log_summary(step, train_metrics, extra_logs=extra_logs, writer=writer, prefix="train")
      train_metrics = []
    
    # Evaluate and store checkpoints.
    if step % cfg.log_eval_steps == 0 or step == cfg.total_steps:
      info("Running eval")
      with progress.timed("eval"):
        eval_metrics = []
        for _ in range(20):
          eval_batch = next(val_iter)
          metrics = eval_step(state, eval_batch)
          eval_metrics.append(jax.device_get(jax_utils.unreplicate(metrics)))
        u.log_summary(step, eval_metrics, writer=writer, prefix="val")

      with progress.timed("checkpoint"):
        if lead_host:
          save_checkpoint(FLAGS.workdir, u.unreplicate_and_get(state), step=step)

    writer.flush()

if __name__ == "__main__":
  app.run(main)
