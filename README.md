# GPT-2 in Jax/Flax

This is a Jax/Flax reimplementation of GPT-2 family of models on FineWeb-Edu dataset, inspired from [karpathy/build_nanoGPT](https://github.com/karpathy/build-nanogpt).

Updates:
- [x] Add support for `tf.data` pipelines over TFRecords.
- [x] Add support for `bfloat16` computation.
- [x] SPMD (multi-node) training support using `pmap`.
- [x] Expose configurables via CLI flags (or config dict).
- [x] Use cuDNN flash attention kernel (SDPA API) (https://github.com/google/jax/issues/22546).
- [x] `nn.Embed` typecast performance issue.
- [x] Use scale init for residual paths.
- [x] Fix large gradient norm spikes for longer training runs.
- [x] Test `accumulate_gradient`.
- [x] Update docstrings.
- [ ] Add `shard_map` support for model and data sharding.
- [ ] KV cache decoding.
### Setup
Create a virtual environment and install packages.
```shell
pip install -r requirements.txt
```

For SPMD support (multi-node training), install OpenMPI.
```shell
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

### Prepare `TFRecords`
```shell
python fineweb.py --outdir /path/to/store/tfrecord
```

### Train
```shell
# Single process, multi-GPU.
python train.py --workdir artifacts/gpt2_124M --config configs/default.py

# multi-process on same host using OpenMPI.
mpirun -n 8 \
          -bind-to socket \
          python train.py --workdir artifacts/gpt2_124M --config configs/default.py

# multi-node across 8 hosts (needs passwordless SSH across hosts).
mpirun -n 8 \
          -pernode \
          -H hostname1,hostname2,...,hostname8 \
          -bind-to socket \
          python train.py --workdir artifacts/gpt2_124M --config configs/default.py
```
<img src="https://github.com/MasterSkepticista/gpt2/raw/main/.github/loss.jpg">

### License
MIT