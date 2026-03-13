"""
Ved — a small language model built for mobile-first inference.

Architecture choices vs vanilla GPT-2:
  - RMSNorm      : faster than LayerNorm (no mean subtraction, no bias)
  - RoPE         : rotary positional embeddings (no learned wpe table)
  - GQA          : grouped-query attention — n_kv_head << n_head,
                   shrinks the KV-cache by n_head/n_kv_head ×
  - SwiGLU       : gated FFN (used in LLaMA/Mistral), empirically better
  - KV Cache     : cached past keys/values for O(1) incremental decoding
  - no bias      : slightly faster + marginally better on large data
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VedConfig:
    block_size: int  = 1024
    vocab_size: int  = 50304   # GPT-2 vocab (50257) padded to next mult-of-64
    n_layer:    int  = 8
    n_head:     int  = 8       # query heads
    n_kv_head:  int  = 2       # key/value heads  (GQA: must divide n_head)
    n_embd:     int  = 512
    dropout:    float = 0.0
    bias:       bool  = False


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square normalisation — faster than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, max_seq: int, theta: float = 10000.0) -> torch.Tensor:
    """Return complex-valued frequency tensor of shape (max_seq, head_dim//2)."""
    freqs   = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t       = torch.arange(max_seq, device=freqs.device)
    freqs   = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(xq: torch.Tensor, xk: torch.Tensor,
               freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K.  Shapes: (B, T, n_heads, head_dim)."""
    def rotate(x):
        x_  = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f   = freqs_cis[: x_.shape[1]].unsqueeze(0).unsqueeze(2)  # (1,T,1,D/2)
        return torch.view_as_real(x_ * f).flatten(3).type_as(x)
    return rotate(xq), rotate(xk)


# ---------------------------------------------------------------------------
# GQA helpers
# ---------------------------------------------------------------------------

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match the number of Q heads."""
    if n_rep == 1:
        return x
    B, T, n_kv, hd = x.shape
    return (
        x.unsqueeze(3)
         .expand(B, T, n_kv, n_rep, hd)
         .reshape(B, T, n_kv * n_rep, hd)
    )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config: VedConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep     = config.n_head // config.n_kv_head
        self.head_dim  = config.n_embd // config.n_head
        self.dropout   = config.dropout

        # Separate Q / K / V projections so K and V can be smaller (GQA)
        self.q_proj  = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=config.bias)
        self.k_proj  = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj  = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj  = nn.Linear(config.n_embd, config.n_embd,                    bias=config.bias)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: slow attention — upgrade to PyTorch >= 2.0 for Flash Attention")

    def forward(
        self,
        x:        torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv:  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        B, T, C = x.shape

        # Project and reshape to (B, T, n_heads, head_dim)
        xq = self.q_proj(x).reshape(B, T, self.n_head,    self.head_dim)
        xk = self.k_proj(x).reshape(B, T, self.n_kv_head, self.head_dim)
        xv = self.v_proj(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE to Q and K
        xq, xk = apply_rope(xq, xk, freqs_cis)

        # Append to KV cache (inference only)
        if past_kv is not None:
            past_k, past_v = past_kv
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)
        present_kv = (xk, xv)

        # Expand KV heads to match Q heads for the attention computation
        xk = repeat_kv(xk, self.n_rep)  # (B, full_T, n_head, head_dim)
        xv = repeat_kv(xv, self.n_rep)

        # (B, n_head, T, head_dim)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Causal masking:
        #   - training (past_kv is None):  full causal mask over T tokens
        #   - inference (past_kv present): single query attends to ALL cached
        #     keys — no mask needed (is_causal=False prevents the wrong mask)
        using_kv_cache = past_kv is not None

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=not using_kv_cache,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att   = (q @ k.transpose(-2, -1)) * scale
            if not using_kv_cache:
                causal = torch.tril(torch.ones(T, k.size(2), device=x.device)).bool()
                att    = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y   = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.o_proj(y))
        return y, present_kv


# ---------------------------------------------------------------------------
# Feed-forward: SwiGLU
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU FFN:  output = down( SiLU(gate(x)) * up(x) )
    Hidden dim is 2/3 × 4 × n_embd, rounded up to next multiple of 256,
    so the total parameter count matches a standard 4× FFN.
    """

    def __init__(self, config: VedConfig):
        super().__init__()
        raw      = int(2 / 3 * 4 * config.n_embd)
        hidden   = 256 * ((raw + 255) // 256)   # align to 256 for hw efficiency

        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.drop      = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, config: VedConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn   = SwiGLU(config)

    def forward(
        self,
        x:         torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_kv = self.attn(self.norm1(x), freqs_cis, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Ved(nn.Module):

    def __init__(self, config: VedConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding ↔ output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE frequencies once; stored as a buffer (moves with .to())
        head_dim = config.n_embd // config.n_head
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(head_dim, config.block_size),
        )

        # Initialise weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            # Scale residual-path output projections (GPT-2 paper §2.3)
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Ved: {self.get_num_params()/1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    def forward(
        self,
        idx:              torch.Tensor,
        targets:          Optional[torch.Tensor] = None,
        past_key_values:  Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        _, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token embeddings
        x = self.transformer.drop(self.transformer.wte(idx))

        # Slice RoPE freqs for the *current* positions
        past_len   = past_key_values[0][0].size(1) if past_key_values is not None else 0
        freqs_cis  = self.freqs_cis[past_len : past_len + T]

        # Forward through transformer blocks
        present_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, freqs_cis, past_kv)
            present_key_values.append(present_kv)

        x = self.transformer.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute logits for the last token
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        return logits, loss, present_key_values

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          Optional[int]   = None,
        top_p:          Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache, top-k, and top-p."""
        past_key_values = None

        for _ in range(max_new_tokens):
            # First step: feed full prompt; subsequent steps: feed only last token
            idx_in = idx if past_key_values is None else idx[:, -1:]

            logits, _, past_key_values = self(idx_in, past_key_values=past_key_values)
            logits = logits[:, -1, :] / temperature

            # top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus)
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens once cumulative prob exceeds top_p
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)

        return idx

    # ------------------------------------------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict    = {n: p for n, p in self.named_parameters() if p.requires_grad}
        # Weight decay only on 2-D tensors (matmul weights), not norms / biases / embeddings
        decay_params  = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params= [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups  = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        n_decay   = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"decayed params: {n_decay:,}  |  non-decayed: {n_nodecay:,}")
        use_fused = device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            fused=use_fused,
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Model FLOP utilisation relative to T4 FP16 peak (65 TFLOPS)."""
        cfg = self.config
        N   = self.get_num_params()
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token   = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd  = flops_per_token * T
        flops_per_iter    = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved    = flops_per_iter / dt
        flops_promised    = 65e12   # T4 FP16 peak
        return flops_achieved / flops_promised
