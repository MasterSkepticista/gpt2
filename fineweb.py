"""FineWeb-Edu TFRecord Dataset (for GPT2 pretraining).
Modified from karpathy/build_nanoGPT.

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Run: python fineweb.py --outdir /path/to/store
"""
import argparse
import os

import numpy as np
import tensorflow as tf
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--outdir", type=str, required=True)
  args = parser.parse_args()

  # Download the dataset.
  ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

  # Tokenizer.
  enc = tiktoken.get_encoding("gpt2")
  eot = enc._special_tokens["<|endoftext|>"]

  def _tokenize(doc):
    # Tokenizes a single document and returns a numpy array of uint16 tokens.
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens = np.array(tokens)
    assert (0 <= tokens).all() and (tokens < 2**16).all(), "Token dictionary too large for uint16"
    # We convert to bytes here to speed up writing records.
    # Also related - https://github.com/huggingface/datasets/issues/4352
    return {"tokens": np.asarray(tokens, dtype=np.uint16).tobytes()}

  ds = ds.map(_tokenize, num_proc=os.cpu_count() // 2, desc="Tokenizing documents")

  # Shard dataset
  num_shards = 100
  shards = [ds.shard(num_shards, i) for i in range(num_shards)]

  # Write TFRecords.
  os.makedirs(args.outdir)
  
  for i, shard in enumerate(shards):  
    split = "val" if i == 0 else "train"
    path = os.path.join(args.outdir, f"{split}_{i:03d}.tfrecord")
    
    with tf.io.TFRecordWriter(path) as writer:
      for example in tqdm(shard, desc=f"Shard {i:03d}"):
        example = tf.train.Example(
          features=tf.train.Features(feature={
            "tokens": tf.train.Feature(bytes_list=tf.train.BytesList(value=[example["tokens"]])),
          }))
        writer.write(example.SerializeToString())

if __name__ == "__main__":
  main()