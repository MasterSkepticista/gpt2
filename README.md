# GPT-2 in Jax/Flax

This is a Jax/Flax reimplementation of GPT-2 family of models on FineWeb-Edu dataset, inspired from [karpathy/build_nanoGPT](https://github.com/karpathy/build-nanogpt).

Updates:
- [x] Add support for `tf.data` pipelines over TFRecords.
- [x] Add support for `bfloat16` computation.
- [ ] Use `cudnn_dot_product_attention`.
- [ ] Expose configurables via CLI flags.
- [x] SPMD (multi-node) training support.

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