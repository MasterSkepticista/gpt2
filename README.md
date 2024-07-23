# GPT-2 in Jax/Flax

This is a Jax/Flax reimplementation of GPT-2 family of models on FineWeb-Edu dataset, inspired from [karpathy/build_nanoGPT](https://github.com/karpathy/build-nanogpt).

Updates:
- [x] Add support for `tf.data` pipelines over TFRecords.
- [x] Add support for `bfloat16` computation.
- [x] SPMD (multi-node) training support using `pmap`.
- [ ] Expose configurables via CLI flags (or config dict).
- [ ] Use cuDNN flash attention kernel.
- [ ] Add `shard_map` support for model and data sharding.
- [ ] `nn.Embed` typecast performance issue.
- [ ] Refactor `load_hf_pretrained` to support split-dense qkv weights.
- [ ] Use scale init for residual paths.
- [ ] Finish incomplete docstrings.
## Setup
Create a virtual environment and install packages.
```shell
$> pip install -r requirements.txt
```

For SPMD support (multi-node training), install OpenMPI.
```shell
$> sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

## Prepare `TFRecords`
```shell
$> python fineweb.py --outdir /path/to/store/tfrecord
```

## Train
```shell
# Single process, multi-GPU.
$> python train.py --workdir artifacts/gpt2_124M

# multi-process on same host using OpenMPI.
$> mpirun -n 8 -bind-to socket python train.py --workdir artifacts/gpt2_124M

# multi-node across 8 hosts (ensure you have common NFS mounts figured out).
$> mpirun -n 8 \
          -pernode \
          -H hostname1,hostname2,...,hostname8 \
          -bind-to socket \
          python train.py --workdir artifacts/gpt2_124M
```

## License
MIT