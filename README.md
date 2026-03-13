# Ved

A small language model built for mobile-first inference.

Ved is a from-scratch transformer with modern architecture choices that shrink the model and KV cache without sacrificing quality — making it practical to run on-device.

## Architecture

| Feature | Choice | Why |
|---------|--------|-----|
| Normalisation | **RMSNorm** | No mean subtraction or bias; faster than LayerNorm |
| Position encoding | **RoPE** | Relative positions, no learned embedding table |
| Attention | **GQA** (grouped-query) | `n_kv_head ≪ n_head` — shrinks KV cache by `n_head/n_kv_head ×` |
| Feed-forward | **SwiGLU** | Gated FFN (LLaMA/Mistral style); empirically better than GELU |
| Inference | **KV Cache** | O(1) incremental decoding — only the new token is processed each step |
| Biases | **None** | Slightly faster; marginally better at scale |

Default config (`VedConfig`): 8 layers · 8 query heads · 2 KV heads · 512 embedding dim → ~50M parameters.

## Install

```bash
pip install torch numpy tiktoken wandb
```

## Data prep

Tokenise any raw text corpus into binary `train.bin` / `val.bin` files.
The simplest way is to use the GPT-2 BPE tokeniser via tiktoken.

Example — Shakespeare character-level (for quick experiments):

```bash
# Download & tokenise
python - <<'EOF'
import requests, tiktoken, numpy as np
text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
enc  = tiktoken.get_encoding("gpt2")
ids  = enc.encode(text)
arr  = np.array(ids, dtype=np.uint16)
n    = int(0.9 * len(arr))
arr[:n].tofile("data/shakespeare/train.bin")
arr[n:].tofile("data/shakespeare/val.bin")
EOF
```

## Training

**Single GPU (local)**

```bash
python train.py \
    --dataset shakespeare \
    --n_layer 6 --n_head 6 --n_kv_head 2 --n_embd 384 \
    --max_iters 5000 --batch_size 32 \
    --wandb_project ved
```

**Resume from checkpoint**

```bash
python train.py --init_from resume --out_dir out
```

**Disable wandb**

```bash
python train.py --no_wandb
```

**Config file** (optional — plain Python, values act as argument defaults)

```bash
python train.py config/train_ved.py
python train.py config/train_ved.py --lr 1e-4   # CLI overrides config file
```

**DDP (multi-GPU)**

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_ved.py
```

## Training on Kaggle (free T4 GPU)

1. Create a new Kaggle notebook and enable **GPU T4 x2** accelerator.
2. In the first cell:
```bash
%%bash
pip install -q tiktoken wandb
git clone https://github.com/orcus108/ved
cd ved
# Prepare data (Shakespeare example)
mkdir -p data/shakespeare
python - <<'EOF'
import requests, tiktoken, numpy as np
text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
enc  = tiktoken.get_encoding("gpt2")
ids  = enc.encode(text)
arr  = np.array(ids, dtype=np.uint16)
n    = int(0.9 * len(arr))
arr[:n].tofile("data/shakespeare/train.bin")
arr[n:].tofile("data/shakespeare/val.bin")
EOF
```
3. In the next cell:
```bash
%%bash
cd ved
python train.py \
    --dataset shakespeare \
    --device cuda \
    --max_iters 10000 \
    --batch_size 64 \
    --gradient_accumulation_steps 4 \
    --wandb_project ved
```
4. Checkpoints are saved to `out/ckpt.pt`. Download via the Kaggle output panel.

## Sampling

```bash
python sample.py \
    --ckpt out/ckpt.pt \
    --prompt "To be, or not to be" \
    --num_samples 3 \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 100
```

Load prompt from a file:

```bash
python sample.py --ckpt out/ckpt.pt --prompt FILE:my_prompt.txt
```

## Files

```
model.py    — Ved model (RMSNorm, RoPE, GQA, SwiGLU, KV Cache)
train.py    — Training loop (argparse config, wandb, gradient accumulation)
sample.py   — Generation CLI (top-k, top-p, KV cache)
```

---

*Architecture inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).*
