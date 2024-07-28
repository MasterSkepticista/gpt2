"""GPT2 124M model.

Trains GPT-2 124M model on a FineWeb-Edu dataset for 1 epoch (10B tokens).
This config achieves a val loss of ~3.10.
"""
import ml_collections


def _set_model(config: dict, variant):
  variants = {
      "gpt2": dict(num_layers=12, num_heads=12, emb_dim=768),  # 124M
      "gpt2-medium": dict(num_layers=24, num_heads=16, emb_dim=1024),  # 350M
      "gpt2-large": dict(num_layers=36, num_heads=20, emb_dim=1280),  # 774M
      "gpt2-xl": dict(num_layers=48, num_heads=25, emb_dim=1600),  # 1558M
  }
  config.model = variants[variant]
  config.model.name = variant
  config.model.vocab_size = 50_257
  config.model.block_size = 1024

def get_config():
  config = ml_collections.ConfigDict()
  config.rng_seed = 42

  # Dataset.
  config.data_dir = "./data"

  # Model
  _set_model(config, "gpt2")
  config.model.dtype = "bfloat16"  # Precision of computation.
  # Only supported in JAX SDPA API.
  config.model.sdpa_implementation = "cudnn"  # "xla" or "cudnn".

  # Optimizer
  config.lr = 6e-4
  config.min_lr = 0.1 * config.lr  # Sustained lr after decay.
  config.warmup_steps = 715
  config.total_steps = 19073
  config.grad_clip_norm = 1.0
  config.optax_name = "adamw"
  config.optax_kwargs = dict(b1=0.9, b2=0.95, weight_decay=0.1)

  # Training
  config.batch_size = 512  # Corresponds to 512 * config.model.block_size tokens per batch.
  config.grad_accum_steps = 1  # Increment in orders of 2 if facing OOM.
  config.log_train_steps = 50
  config.log_eval_steps = 1000  # Also checkpoints the model.

  return config