"""Utils."""
import collections
from typing import Callable

import jax


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