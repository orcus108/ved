"""
train.py — Ved training script

Usage:
    python train.py                            # all defaults
    python train.py config/train_ved.py        # load config file
    python train.py config/train_ved.py --lr 3e-4  # config + CLI overrides

DDP:
    torchrun --standalone --nproc_per_node=4 train.py [config] [overrides]
"""

import argparse
import importlib.util
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Ved, VedConfig

# ---------------------------------------------------------------------------
# Argument parsing with optional config-file override
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Train a Ved language model")

    # Optional positional config file
    p.add_argument("config", nargs="?", default=None,
                   help="Python config file (e.g. config/train_ved.py)")

    # I/O
    p.add_argument("--out_dir",                 default="out")
    p.add_argument("--eval_interval",           type=int,   default=500)
    p.add_argument("--log_interval",            type=int,   default=10)
    p.add_argument("--eval_iters",              type=int,   default=200)
    p.add_argument("--eval_only",               action="store_true")
    p.add_argument("--always_save_checkpoint",  action="store_true", default=True)
    p.add_argument("--init_from",               default="scratch",
                   choices=["scratch", "resume"])

    # wandb
    p.add_argument("--wandb_log",       action="store_true", default=True)
    p.add_argument("--no_wandb",        action="store_true",
                   help="Disable wandb logging")
    p.add_argument("--wandb_project",   default="ved")
    p.add_argument("--wandb_run_name",  default="ved-" + str(int(time.time())))

    # Data
    p.add_argument("--dataset",                     default="openwebtext")
    p.add_argument("--gradient_accumulation_steps", type=int,   default=8)
    p.add_argument("--batch_size",                  type=int,   default=16)
    p.add_argument("--block_size",                  type=int,   default=1024)

    # Model
    p.add_argument("--n_layer",     type=int,   default=8)
    p.add_argument("--n_head",      type=int,   default=8)
    p.add_argument("--n_kv_head",   type=int,   default=2)
    p.add_argument("--n_embd",      type=int,   default=512)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--bias",        action="store_true", default=False)

    # Optimiser
    p.add_argument("--lr",              dest="learning_rate", type=float, default=3e-4)
    p.add_argument("--max_iters",       type=int,   default=100_000)
    p.add_argument("--weight_decay",    type=float, default=0.1)
    p.add_argument("--beta1",           type=float, default=0.9)
    p.add_argument("--beta2",           type=float, default=0.95)
    p.add_argument("--grad_clip",       type=float, default=1.0)

    # LR schedule
    p.add_argument("--decay_lr",        action="store_true", default=True)
    p.add_argument("--warmup_iters",    type=int,   default=1000)
    p.add_argument("--lr_decay_iters",  type=int,   default=100_000)
    p.add_argument("--min_lr",          type=float, default=3e-5)

    # DDP
    p.add_argument("--backend", default="nccl")

    # System
    p.add_argument("--device",  default="cuda")
    p.add_argument("--dtype",   default=None,
                   choices=["float32", "bfloat16", "float16"],
                   help="Mixed-precision dtype (default: bfloat16 if supported, else float16)")
    p.add_argument("--compile", action="store_true", default=False)

    return p


def load_config_file(path: str) -> dict:
    """Load a Python config file and return its public namespace."""
    spec = importlib.util.spec_from_file_location("_ved_config", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: v for k, v in vars(mod).items() if not k.startswith("_")}


def parse_args():
    parser = build_parser()

    # First pass: grab only the positional config file (ignore unknown args)
    partial, remaining = parser.parse_known_args()

    if partial.config is not None:
        file_cfg = load_config_file(partial.config)
        # Apply config-file values as new defaults
        for action in parser._actions:
            dest = action.dest
            if dest in file_cfg:
                action.default = file_cfg[dest]

    # Second pass: full parse (remaining CLI args override config file)
    args = parser.parse_args([partial.config] + remaining if partial.config else remaining)

    # --no_wandb overrides --wandb_log
    if args.no_wandb:
        args.wandb_log = False

    # Auto-select dtype
    if args.dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            args.dtype = "bfloat16"
        else:
            args.dtype = "float16"

    return args


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    fname = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([torch.from_numpy(data[i    : i + block_size    ].astype(np.int64)) for i in ix])
    y    = torch.stack([torch.from_numpy(data[i + 1: i + block_size + 1].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff       = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- DDP setup ---
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank       = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device         = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset    = ddp_rank
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        device         = args.device
        master_process = True
        seed_offset    = 0
        ddp_world_size = 1

    tokens_per_iter = (
        args.gradient_accumulation_steps * ddp_world_size
        * args.batch_size * args.block_size
    )
    if master_process:
        print(f"tokens per iteration: {tokens_per_iter:,}")
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype     = {"float32": torch.float32,
                   "bfloat16": torch.bfloat16,
                   "float16":  torch.float16}[args.dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    # --- Data ---
    data_dir = os.path.join("data", args.dataset)

    def _get_batch(split):
        return get_batch(split, data_dir, args.block_size, args.batch_size,
                         device, device_type)

    # --- Vocab size ---
    meta_vocab_size = None
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size}")

    # --- Model init ---
    model_args = dict(
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_kv_head  = args.n_kv_head,
        n_embd     = args.n_embd,
        block_size = args.block_size,
        bias       = args.bias,
        vocab_size = None,
        dropout    = args.dropout,
    )

    iter_num      = 0
    best_val_loss = 1e9

    if args.init_from == "scratch":
        print("Initialising model from scratch")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size else 50304
        model = Ved(VedConfig(**model_args))

    elif args.init_from == "resume":
        ckpt_path  = os.path.join(args.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        for k in ["n_layer", "n_head", "n_kv_head", "n_embd",
                  "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint["model_args"][k]
        model = Ved(VedConfig(**model_args))
        state_dict = checkpoint["model"]
        # Strip torch.compile prefix if present
        for k in list(state_dict):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num      = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from iter {iter_num}, best val loss {best_val_loss:.4f}")

    if args.block_size < model.config.block_size:
        model.config.block_size = args.block_size

    model.to(device)

    # --- Optimiser ---
    scaler    = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    if args.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free memory

    # --- Compile ---
    if args.compile:
        print("Compiling model (takes ~1 min)…")
        model = torch.compile(model)

    # --- DDP wrap ---
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    # --- Loss estimation ---
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = _get_batch(split)
                with ctx:
                    _, loss, _ = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # --- wandb ---
    if args.wandb_log and master_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    # --- Training loop ---
    X, Y = _get_batch("train")
    t0              = time.time()
    local_iter_num  = 0
    running_mfu     = -1.0

    while True:
        # LR schedule
        lr = (get_lr(iter_num, args.warmup_iters, args.lr_decay_iters,
                     args.learning_rate, args.min_lr)
              if args.decay_lr else args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval + checkpoint
        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
            if args.wandb_log:
                import wandb
                wandb.log({
                    "iter":        iter_num,
                    "train/loss":  losses["train"],
                    "val/loss":    losses["val"],
                    "lr":          lr,
                    "mfu":         running_mfu * 100,
                })
            if losses["val"] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    ckpt = {
                        "model":       raw_model.state_dict(),
                        "optimizer":   optimizer.state_dict(),
                        "model_args":  model_args,
                        "iter_num":    iter_num,
                        "best_val_loss": best_val_loss,
                        "config":      vars(args),
                    }
                    torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))
                    print(f"checkpoint saved to {args.out_dir}")

        if iter_num == 0 and args.eval_only:
            break

        # Forward / backward with gradient accumulation
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            with ctx:
                _, loss, _ = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            X, Y = _get_batch("train")
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Timing + logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            lossf = loss.item() * args.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    args.batch_size * args.gradient_accumulation_steps, dt
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, "
                  f"time {dt*1000:.0f}ms, mfu {running_mfu*100:.2f}%")

        iter_num       += 1
        local_iter_num += 1

        if iter_num > args.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
