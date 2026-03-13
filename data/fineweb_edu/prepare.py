"""
Prepare FineWeb-Edu for Ved training — streaming edition.

Streams the dataset directly from HuggingFace without downloading the full
~25 GB corpus to disk. Writes only as many tokens as you need.

Default budget fits comfortably on Kaggle (~20 GB disk):
    TRAIN_TOKENS = 2_000_000_000  → ~4 GB as uint16
    VAL_TOKENS   =    10_000_000  → ~20 MB

Usage (run from the repo root):
    python data/fineweb_edu/prepare.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_TOKENS = 2_000_000_000   # 2B tokens  (~4 GB, fits on Kaggle)
VAL_TOKENS   =    10_000_000   # 10M tokens (~20 MB)
WRITE_EVERY  =   100_000_000   # flush to disk every 100M tokens
# ---------------------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

DATA_DIR = os.path.dirname(__file__)


def stream_to_bin(split_name: str, target_tokens: int):
    out_path = os.path.join(DATA_DIR, f"{split_name}.bin")

    if os.path.exists(out_path):
        existing = os.path.getsize(out_path) // 2  # uint16 = 2 bytes
        if existing >= target_tokens:
            print(f"{split_name}: already have {existing:,} tokens, skipping.")
            return
        else:
            print(f"{split_name}: found {existing:,} tokens, re-generating.")
            os.remove(out_path)

    print(f"\nStreaming FineWeb-Edu → {split_name}.bin  (target: {target_tokens:,} tokens)")

    # Stream directly — no parquet files written to disk
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # For val, skip the first TRAIN_TOKENS-worth of documents so it's held out
    if split_name == "val":
        # ~350 tokens/doc on average; skip roughly that many docs
        skip_docs = TRAIN_TOKENS // 350
        ds = ds.skip(skip_docs)

    buf          = []
    total_written = 0

    with tqdm(total=target_tokens, unit="tok", unit_scale=True) as pbar:
        for doc in ds:
            tokens = [EOT] + enc.encode_ordinary(doc["text"])
            buf.extend(tokens)

            if len(buf) >= WRITE_EVERY:
                chunk = np.array(buf[:WRITE_EVERY], dtype=np.uint16)
                with open(out_path, "ab") as f:
                    chunk.tofile(f)
                total_written += WRITE_EVERY
                pbar.update(WRITE_EVERY)
                buf = buf[WRITE_EVERY:]

                if total_written >= target_tokens:
                    break

    # Flush remainder (up to target)
    remaining = min(len(buf), target_tokens - total_written)
    if remaining > 0:
        chunk = np.array(buf[:remaining], dtype=np.uint16)
        with open(out_path, "ab") as f:
            chunk.tofile(f)
        total_written += remaining
        pbar.update(remaining)

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"{split_name}: {total_written:,} tokens written ({size_gb:.2f} GB)")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    stream_to_bin("train", TRAIN_TOKENS)
    stream_to_bin("val",   VAL_TOKENS)
    print("\nDone. Run training with:")
    print("  python train.py config/train_ved.py")


if __name__ == "__main__":
    main()
