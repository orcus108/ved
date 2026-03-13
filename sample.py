"""
sample.py — generate text from a trained Ved checkpoint

Usage:
    python sample.py --ckpt out/ckpt.pt --prompt "Once upon a time"
    python sample.py --ckpt out/ckpt.pt --prompt "FILE:prompt.txt" --num_samples 5
"""

import argparse
import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model import Ved, VedConfig


def parse_args():
    p = argparse.ArgumentParser(description="Sample from a trained Ved model")

    p.add_argument("--ckpt",           required=True,
                   help="Path to checkpoint file (e.g. out/ckpt.pt)")
    p.add_argument("--prompt",         default="\n",
                   help="Prompt text, or FILE:<path> to load from a file")
    p.add_argument("--num_samples",    type=int,   default=3)
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--temperature",    type=float, default=0.8,
                   help="Sampling temperature (<1 = sharper, >1 = more random)")
    p.add_argument("--top_k",          type=int,   default=200)
    p.add_argument("--top_p",          type=float, default=None,
                   help="Nucleus sampling probability (optional)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--dtype",          default=None,
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compile",        action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    device      = args.device
    device_type = "cuda" if "cuda" in device else "cpu"

    if args.dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            args.dtype = "bfloat16"
        else:
            args.dtype = "float16"

    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16":  torch.float16}[args.dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    # --- Load checkpoint ---
    print(f"Loading checkpoint from {args.ckpt}…")
    checkpoint = torch.load(args.ckpt, map_location=device)

    config = VedConfig(**checkpoint["model_args"])
    model  = Ved(config)

    state_dict = checkpoint["model"]
    for k in list(state_dict):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    if args.compile:
        model = torch.compile(model)

    # --- Tokeniser ---
    # Use the dataset-specific vocab if available, otherwise fall back to GPT-2 BPE
    encode = decode = None

    train_cfg = checkpoint.get("config", {})
    dataset   = train_cfg.get("dataset", None)
    if dataset is not None:
        meta_path = os.path.join("data", dataset, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            stoi, itos = meta["stoi"], meta["itos"]
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: "".join(itos[i] for i in l)
            print(f"Using character-level vocab from {meta_path}")

    if encode is None:
        print("Using GPT-2 BPE tokeniser (tiktoken)")
        enc    = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # --- Prompt ---
    prompt = args.prompt
    if prompt.startswith("FILE:"):
        with open(prompt[5:], "r", encoding="utf-8") as f:
            prompt = f.read()

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # --- Generate ---
    print(f"\nGenerating {args.num_samples} sample(s)…\n")
    print("=" * 60)
    with torch.no_grad():
        with ctx:
            for i in range(args.num_samples):
                y = model.generate(
                    x,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print(decode(y[0].tolist()))
                if i < args.num_samples - 1:
                    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    main()
