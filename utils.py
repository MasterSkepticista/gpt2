"""Utils."""
import collections
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
from absl import logging
from clu import metric_writers

PyTree = Any


def compute_flops(apply_fn: Callable,
                  dummy_inputs: list,
                  fuse_multiply_add: bool = True) -> float:
  """Compute the number of FLOPs of a Flax model."""
  analysis = jax.jit(apply_fn).lower(*dummy_inputs).compile().cost_analysis()[0]
  # Not all JAX backends return analysis.
  # See: https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available
  # Ideally we should be able to get flops analysis with `CPU` backend to save GPU memory.
  flops = 0 if analysis["flops"] == -1.0 else analysis["flops"]
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

def unreplicate_and_get(tree: PyTree) -> PyTree:
  """Fetches to CPU the first local copy of a `pmap` replicated tree."""
  return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], tree))

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
                extra_logs: dict = None,
                writer: metric_writers.MetricWriter = None,
                prefix: str = "train"):
  """Logs train summary and optionally writes summaries.
  
  Args:
    step: Integer, current step.
    metrics: A list of metric dictionaries collected over steps.
    extra_logs: A dict of addl. logs (e.g. learning rates).
    writer: Optional metric writer to write summaries to a file.
    prefix: Prefix to be applied on metric keys.
  
  Returns:
    Nothing.
  """
  # Transpose: list of dicts to dict of lists.
  metrics = jax.tree.map(lambda *vals: jnp.stack(vals), *metrics)

  # Log only on main host.
  if jax.process_index() == 0:
    summaries = extra_logs or {}
    summaries.update({
        "/".join((prefix, key)): val.mean() for key, val in metrics.items()
    })

    # Log to stdout
    for name, value in summaries.items():
      logging.info(f"\u001b[35m[{step}]\u001b[0m {name}={float(value):.5f}")

    if writer is not None:
      writer.write_scalars(step, summaries)


def accumulate_gradient(value_and_grad_fn,
                        params: PyTree,
                        batch: PyTree,
                        accum_steps: int = 1) -> Tuple[jnp.ndarray, PyTree]:
  """Accumulates gradients over given steps.
  
  Args:
    value_and_grad_fn: Gradient function that does not return aux values.
    params: Parameters, passed as first argument to `value_and_grad_fn`.
    batch: Batch, passed as second argument to `value_and_grad_fn`.
    accum_steps: Number of micro batches to accumulate over. Defaults to 1,
      which means no gradients are accumulated.
  
  Returns:
    Tuple (loss, grads).
  """
  if accum_steps > 1:
    bs = next(iter(jax.tree.leaves(batch))).shape[0]
    assert bs % accum_steps == 0, (
        f"Invalid accum_steps {accum_steps} for batch size `{bs}")
    microbatch_size = bs // accum_steps
    logging.info("Accumulating with microbatch_size %d over %d steps.",
                 microbatch_size, accum_steps)

    def get_microbatch(batch, i):
      return jax.tree.map(
          lambda t: jnp.reshape(t, (accum_steps, -1) + (t.shape[1:]))[i], batch)

    # Initialize accumulator.
    l, g = value_and_grad_fn(params, get_microbatch(batch, 0))

    def accumulate(i, l_and_g):
      l, g = l_and_g
      l_i, g_i = value_and_grad_fn(params, get_microbatch(batch, i))
      return (l + l_i, jax.tree.map(jnp.add, g, g_i))

    # Average over accum_steps.
    loss, grads = jax.lax.fori_loop(1, accum_steps, accumulate, (l, g))
    return jax.tree.map(lambda x: x / accum_steps, (loss, grads))
  else:
    return value_and_grad_fn(params, batch)
