"""Utils."""
import collections
from typing import Any, Callable

import jax
import jax.numpy as jnp

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


@jax.jit
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
    if step < warmup_steps:
      return max_lr * step / warmup_steps
    if step > max_steps:
      return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1.0
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + jnp.cos(decay_ratio * jnp.pi))

  return sched_fn
