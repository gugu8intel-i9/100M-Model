#!/usr/bin/env python3
"""
=============================================================================
Linear Attention  |  Full Attention  |  Mamba-3 Block
=============================================================================
Drop-in nn.Module blocks for the 100M-Model codebase.
Each block accepts (B, T, C) and returns (B, T, C).
Implemented in pure PyTorch -- no mamba-ssm dependency.
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BlockConfig:
    hidden_dim: int = 768
    num_heads: int = 12
    mlp_ratio: int = 4
    max_seq_len: int = 2048
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, inner_dim, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inner_dim, bias=False)
        self.up_proj   = nn.Linear(dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, dim, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


# ############################################################################
# 1. FULL ATTENTION
# ############################################################################
class FullAttention(nn.Module):
    def __init__(self, cfg, use_ffn=True):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.head_dim
        self.scale     = self.head_dim ** -0.5
        self.hidden_dim = cfg.hidden_dim
        self.use_ffn   = use_ffn
        self.qkv_proj = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(cfg.max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        self.register_buffer("causal_mask", torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0))
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)
        self.attn_norm = RMSNorm(cfg.hidden_dim)
        if use_ffn:
            self.ffn      = SwiGLU(cfg.hidden_dim, cfg.hidden_dim * cfg.mlp_ratio, cfg.dropout)
            self.ffn_norm = RMSNorm(cfg.hidden_dim)

    def _apply_rope(self, x):
        T = x.shape[2]
        x = x.float().reshape(*x.shape[:-1], -1, 2)
        half = x.shape[-2]
        cos = self.cos_cached[:T, :half].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        sin = self.sin_cached[:T, :half].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        x0, x1 = x[..., 0], x[..., 1]
        out = torch.stack([x0*cos.squeeze(-1) - x1*sin.squeeze(-1),
                            x0*sin.squeeze(-1) + x1*cos.squeeze(-1)], dim=-1)
        return out.flatten(-2).type_as(x)

    def forward(self, x):
        B, T, C = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        q = self._apply_rope(q); k = self._apply_rope(k)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(B, T, C)
        x = x + self.proj_drop(self.out_proj(out))
        if self.use_ffn: x = x + self.ffn(self.ffn_norm(x))
        return x


# ############################################################################
# 2. LINEAR ATTENTION
# ############################################################################
class LinearAttention(nn.Module):
    def __init__(self, cfg, use_ffn=True):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.head_dim
        self.hidden_dim = cfg.hidden_dim
        self.use_ffn   = use_ffn
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.proj_drop = nn.Dropout(cfg.dropout)
        self.attn_norm = RMSNorm(cfg.hidden_dim)
        if use_ffn:
            self.ffn      = SwiGLU(cfg.hidden_dim, cfg.hidden_dim * cfg.mlp_ratio, cfg.dropout)
            self.ffn_norm = RMSNorm(cfg.hidden_dim)

    @staticmethod
    def _phi(x): return F.elu(x) + 1.0

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        h = self.attn_norm(x)
        q = self._phi(self.q_proj(h).view(B,T,H,D).transpose(1,2))
        k = self._phi(self.k_proj(h).view(B,T,H,D).transpose(1,2))
        v = self.v_proj(h).view(B,T,H,D).transpose(1,2)
        outer = k.unsqueeze(-1) * v.unsqueeze(-2)
        num_s = torch.cumsum(outer, dim=2)
        den_s = torch.cumsum(k, dim=2)
        num = torch.einsum("bhtd,bhtdd->bhtd", q, num_s)
        den = torch.einsum("bhtd,bhtd->bht", q, den_s).unsqueeze(-1).clamp(min=1e-6)
        out = (num / den).transpose(1,2).reshape(B, T, C)
        x = x + self.proj_drop(self.out_proj(out))
        if self.use_ffn: x = x + self.ffn(self.ffn_norm(x))
        return x


# ############################################################################
# 3. MAMBA-3 BLOCK -- SSD, trapezoidal discretisation, inference-first
# ############################################################################
class Mamba3SSDCore(nn.Module):
    r"""Mamba-3 SSM core (Lahoti et al., 2026).

    SSD framework + trapezoidal discretisation + group-based (N,N) A matrices.

    vs Mamba-1:
      - Trapezoidal rule (2nd-order) instead of zero-order hold
      - Structured (N,N) A per group (not per-channel diagonal)
      - State size = N (head_dim), not a separate d_state
    """

    def __init__(self, d_model, d_conv=4, expand=2, head_dim=64,
                 dt_rank="auto", dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = d_model * expand
        self.N       = head_dim
        self.G       = self.d_inner // self.N
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d  = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  padding=d_conv-1, groups=self.d_inner, bias=True)
        self.dt_proj_bias = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        self.dt_proj      = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # Structured A: (G, N, N) negative diagonal
        A = torch.zeros(self.G, self.N, self.N)
        for g in range(self.G):
            d = torch.arange(1, self.N+1, dtype=torch.float32)
            A[g] = torch.diag(-torch.exp(d * math.log(1 + 1.0/self.N)))
        self.A_log = nn.Parameter(torch.log(-A + 1e-8))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        G, N = self.G, self.N
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        xc = x_inner.transpose(1,2)
        xc = self.conv1d(xc)[:,:,:T].transpose(1,2)
        xc = F.silu(xc)

        dt   = self.dt_proj(F.softplus(self.dt_proj_bias(xc)))
        ssm_B = self.B_proj(xc)
        ssm_C = self.C_proj(xc)

        dt_g = dt.view(B, T, G, N)
        B_g  = ssm_B.view(B, T, G, N)
        C_g  = ssm_C.view(B, T, G, N)

        # Precompute A (fixed per group)
        A = -torch.exp(self.A_log)  # (G, N, N)
        I_mat = torch.eye(N, device=x.device, dtype=x.dtype)

        # Recurrent scan with per-step trapezoidal discretisation
        # (avoids materialising T x G x N x N intermediate tensors)
        h = torch.zeros((B, G, N), device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt_g[:, t].mean(dim=-1, keepdim=True)  # (B, G, 1)
            B_t  = B_g[:, t]                                # (B, G, N)
            C_t  = C_g[:, t]                                # (B, G, N)

            # Trapezoidal discretisation for this step:
            #   dA = dt_t * A   (B, G, N, N) via broadcast
            #   A_bar = (I + dA/2)^{-1} (I - dA/2)
            #   B_bar = (I + dA/2)^{-1} dt_t B_t
            dA = dt_t.unsqueeze(-1) * A.unsqueeze(0)  # (B,G,1,1) * (1,G,N,N) = (B,G,N,N)
            Ip = I_mat + dA * 0.5
            Im = I_mat - dA * 0.5
            A_bar = torch.linalg.solve(Ip, Im)  # (B,G,N,N)
            B_bar = torch.linalg.solve(Ip, (dt_t * B_t).unsqueeze(-1)).squeeze(-1)  # (B,G,N)

            h = torch.einsum('bgmn,bgm->bgn', A_bar, h) + B_bar
            ys.append(h * C_t)

        y = torch.stack(ys, dim=1).reshape(B, T, self.d_inner)
        y = y + self.D * xc
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


class MambaBlock(nn.Module):
    """Mamba-3 block: (B,T,C) -> (B,T,C)."""

    def __init__(self, cfg, d_conv=4, expand=2, use_ffn=True):
        super().__init__()
        self.use_ffn = use_ffn
        self.norm = RMSNorm(cfg.hidden_dim)
        self.ssm  = Mamba3SSDCore(d_model=cfg.hidden_dim, d_conv=d_conv,
                                   expand=expand, head_dim=cfg.head_dim,
                                   dropout=cfg.dropout)
        if use_ffn:
            self.ffn      = SwiGLU(cfg.hidden_dim, cfg.hidden_dim*cfg.mlp_ratio, cfg.dropout)
            self.ffn_norm = RMSNorm(cfg.hidden_dim)

    def forward(self, x):
        x = x + self.ssm(self.norm(x))
        if self.use_ffn: x = x + self.ffn(self.ffn_norm(x))
        return x


# ############################################################################
# 4. HYBRID MODEL
# ############################################################################
class HybridModel(nn.Module):
    BLOCK_REGISTRY = {"full": FullAttention, "linear": LinearAttention, "mamba": MambaBlock}

    def __init__(self, cfg, vocab_size=32000, layer_schedule=None):
        super().__init__()
        self.cfg = cfg
        if layer_schedule is None:
            layer_schedule = ["full","full","linear","mamba","linear","mamba"]
        self.layer_schedule = layer_schedule
        self.tok_embeddings = nn.Embedding(vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([self.BLOCK_REGISTRY[bt](cfg) for bt in layer_schedule])
        self.norm  = RMSNorm(cfg.hidden_dim)
        self.output = nn.Linear(cfg.hidden_dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        n = sum(p.numel() for p in self.parameters())
        print(f"HybridModel: {n:,} params ({n/1e6:.2f}M)")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, targets=None):
        x = self.tok_embeddings(input_ids)
        for layer in self.layers: x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


def benchmark_blocks():
    import time
    cfg = BlockConfig(hidden_dim=768, num_heads=12, max_seq_len=512, dropout=0.0)
    B, T, C = 2, 128, cfg.hidden_dim
    x = torch.randn(B, T, C)
    targets = torch.randint(0, 1000, (B, T))
    input_ids = torch.randint(0, 1000, (B, T))
    blocks = {"FullAttention": FullAttention(cfg, use_ffn=True),
              "LinearAttention": LinearAttention(cfg, use_ffn=True),
              "Mamba-3": MambaBlock(cfg, use_ffn=True)}
    print("="*70)
    print(f"  Benchmark | B={B} T={T} C={C} H={cfg.num_heads} D={cfg.head_dim}")
    print("="*70)
    for name, blk in blocks.items():
        np_ = sum(p.numel() for p in blk.parameters())
        _ = blk(x)
        t0 = time.perf_counter(); out = blk(x); t1 = time.perf_counter()
        print(f"  {name:20s} | params={np_:>10,} | out={str(list(out.shape)):>14s} | {(t1-t0)*1000:>7.2f} ms")
    print("-"*70)
    hybrid = HybridModel(cfg, vocab_size=1000,
                          layer_schedule=["full","linear","mamba","full","mamba","linear"])
    logits, loss = hybrid(input_ids, targets)
    print(f"  HybridModel (e2e) | logits={list(logits.shape)} | loss={loss.item():.4f}")
    print("="*70)
    print("  All blocks OK!")


if __name__ == "__main__":
    benchmark_blocks()
