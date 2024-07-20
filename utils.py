"""Utils."""
import collections
from typing import Any, Callable, List

import jax
import jax.numpy as jnp
from absl import logging
from clu import metric_writers

PyTree = Any


def compute_flops(apply_fn: Callable,
                  dummy_inputs: list,
                  fuse_multiply_add: bool = True) -> float:
  """Compute the number of FLOPs of a Flax model."""
  analysis = jax.jit(
      apply_fn, backend="cpu").lower(*dummy_inputs).cost_analysis()
  flops = analysis["flops"]
  if fuse_multiply_add:
    flops = flops / 2
  return flops


def recover_tree(keys, values, sep: str = "."):
  """Unflatten key-value pairs to a nested dictionary where each key is `sep` path separated."""
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if sep not in k:
      tree[k] = v
    else:
      left, right = k.split(sep, 1)
      sub_trees[left].append((right, v))
  for k, kv_pairs in sub_trees.items():
    tree[k] = recover_tree(*zip(*kv_pairs))
  return tree


def tf_to_numpy(batch: PyTree) -> PyTree:
  """Zero-copy numpy conversion."""
  return jax.tree.map(lambda x: x._numpy(), batch)


def shard_batches(batch: PyTree, num_devices: int = None) -> PyTree:
  """Shard batch to `num_devices` or as inferred from local device count."""
  num_devices = num_devices or jax.local_device_count()
  return jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]),
                      batch)


def get_cosine_lr_schedule(max_lr: float, min_lr: float, max_steps: int,
                           warmup_steps: int) -> Callable[[int], float]:
  """Cosine learning rate schedule.
  
  Args:
    max_lr: Peak learning rate.
    min_lr: Minimum constant learning rate after cosine decay.
    max_steps: Number of steps to decay over the entire training (including warmup).
    warmup_steps: Number of steps to linearly increase learning rate from 0 to `max_lr`.
  
  Returns:
    A function that returns lr for requested step.
  """

  def sched_fn(step: int) -> float:
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = jnp.clip(decay_ratio, 0.0, 1.0)
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + jnp.cos(decay_ratio * jnp.pi))
    lr = jnp.minimum(lr, max_lr * step / warmup_steps)
    return lr

  return sched_fn


def log_summary(step: int,
                metrics: List[dict],
                extra_logs: dict,
                writer: metric_writers.MetricWriter = None,
                prefix: str = "train"):
  """Logs train summary and optionally writes summaries.
  
  Args:
    metrics: A list of metric dictionaries collected over steps.
    writer: Optional metric writer to write summaries to a file.
  """
  # Transpose: list of dicts to dict of lists.
  metrics = jax.tree.map(lambda *vals: jnp.stack(vals), *metrics)
  
  # Log only on main host.
  if jax.process_index() == 0:
    summaries = extra_logs
    summaries.update({
        "/".join((prefix, key)): val.mean()
        for key, val in metrics.items()
    })

    # Log to stdout
    for name, value in summaries.items():
      logging.info(f"\u001b[35m[{step}]\u001b[0m {name}={float(value):.5f}")

    if writer is not None:
      writer.write_scalars(step, summaries)
      writer.flush()