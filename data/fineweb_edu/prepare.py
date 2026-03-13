"""
Prepare FineWeb-Edu (sample-10BT) for Ved training.

Downloads ~10B tokens of high-quality educational web text from HuggingFace,
tokenises with the GPT-2 BPE tokeniser (tiktoken), and writes:
    data/fineweb_edu/train.bin   — uint16 token ids
    data/fineweb_edu/val.bin     — uint16 token ids

Usage (run from the repo root):
    python data/fineweb_edu/prepare.py

Args you can override at the top of the file:
    NUM_PROC       — parallel workers for tokenisation
    VAL_SIZE       — number of documents held out for validation
    SHARD_SIZE     — tokens per shard written to disk

The script streams the dataset so it never loads everything into RAM at once.
On a Kaggle T4 notebook this takes ~30-60 minutes depending on bandwidth.
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_PROC   = 4        # tokenisation workers
VAL_SIZE   = 5_000   # documents reserved for validation
SHARD_SIZE = int(1e8) # ~100M tokens per shard before flushing to disk
# ---------------------------------------------------------------------------

enc      = tiktoken.get_encoding("gpt2")
EOT      = enc._special_tokens["<|endoftext|>"]  # document separator

DATA_DIR = os.path.dirname(__file__)


def tokenise(doc):
    tokens = [EOT]  # prepend EOT so every doc starts with a separator
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return {"ids": tokens, "len": len(tokens)}


def write_bin(path, all_tokens):
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(path)
    print(f"wrote {len(arr):,} tokens → {path}")


def main():
    print("Loading FineWeb-Edu (sample-10BT) in streaming mode …")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=False,   # full download — Kaggle has plenty of disk
        num_proc=NUM_PROC,
    )

    # Tokenise in parallel
    print("Tokenising …")
    tokenised = ds.map(
        tokenise,
        remove_columns=ds.column_names,
        desc="tokenising",
        num_proc=NUM_PROC,
    )

    # Train / val split
    split = tokenised.train_test_split(test_size=VAL_SIZE, seed=42)

    for name, subset in [("train", split["train"]), ("val", split["test"])]:
        out_path = os.path.join(DATA_DIR, f"{name}.bin")
        all_tokens = []
        total = 0
        for sample in tqdm(subset, desc=name):
            all_tokens.extend(sample["ids"])
            total += sample["len"]
            if len(all_tokens) >= SHARD_SIZE:
                # flush shard
                arr = np.array(all_tokens, dtype=np.uint16)
                with open(out_path, "ab") as f:
                    arr.tofile(f)
                all_tokens = []
        if all_tokens:
            arr = np.array(all_tokens, dtype=np.uint16)
            with open(out_path, "ab") as f:
                arr.tofile(f)
        print(f"{name}: {total:,} tokens")

    print("Done. Run training with:")
    print("  python train.py config/train_ved.py")


if __name__ == "__main__":
    main()
