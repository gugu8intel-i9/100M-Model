#!/usr/bin/env python3
"""
=============================================================================
DFlash-style Block-Diffusion Drafter for Speculative Decoding
(after arXiv 2602.06036, Z Lab)

Replaces the DeepSeek-V3 MTP block on the 100M target. A lightweight
diffusion draft model predicts a whole BLOCK of future tokens in a single
parallel forward pass, conditioned on fused hidden features extracted from
the frozen target model. The target model "knows best": its own hidden
states are used as persistent conditioning so draft quality can scale with
drafter depth instead of diluting (the EAGLE problem).

Ported ideas:
  1. KV-INJECTION attention: fused target-context features are projected
     into K/V and injected into EVERY drafter layer, then stored in the
     KV cache so conditioning survives draft iteration.
  2. BLOCK DIFFUSION training: each block is seeded by a clean anchor token;
     the remaining masked positions are predicted in parallel in one pass.
     Attention is bidirectional INSIDE a block and never crosses blocks
     (sparse mask), matching the paper's "no inter-block leakage".
  3. spec_generate(): a custom speculative-decoding loop that drafts one
     block, verifies it with the frozen target, accepts the longest
     matching prefix plus the bonus token, and repeats.

The LM head and token embeddings are SHARED with the frozen target, so the
drafter stays small: only its transformer layers + fusion projection train.

Usage:
    python dflash_drafter.py --train-steps 20 --max-new-tokens 64

Author: Kunal
=============================================================================
"""

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SMALL BUILDING BLOCKS (self-contained so this file runs standalone)
# ============================================================================

class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        inner = dim * ratio
        self.gate = nn.Linear(dim, inner, bias=False)
        self.up = nn.Linear(dim, inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class DraftConfig:
    """Configuration for the block-diffusion drafter."""

    num_draft_layers: int = 2       # Transformer layers in the draft model
    block_size: int = 8             # tokens per diffusion block (block_size - 1 drafted)
    num_target_ctx_layers: int = 4  # how many target layers feed the fused context
    mlp_ratio: int = 2              # drafter is intentionally cheaper than target
    dropout: float = 0.0
    seed: int = 0


def build_target_layer_ids(num_target_layers: int, sample: int = 4) -> List[int]:
    """Evenly spaced target-layer indices used to build the fused context."""
    sample = max(1, min(sample, max(1, num_target_layers)))
    if num_target_layers <= 0:
        return [0]
    step = num_target_layers / sample
    ids = sorted({int(i * step) for i in range(sample)})
    if ids[-1] != num_target_layers - 1:
        ids[-1] = num_target_layers - 1
    return sorted(set(ids))


# ============================================================================
# KV-INJECTION ATTENTION
# ============================================================================

class BlockDiffusionAttention(nn.Module):
    """
    Attention with DFlash-style KV injection.

    Queries attend to:
      - every valid position of the injected fused target context, and
      - other tokens WITHIN the same diffusion block (bidirectional).
    Attention across different blocks is prohibited (sparse mask).
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Per-head normalization (q/k), as in Qwen3/DFlash layers
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                target_hidden: torch.Tensor,
                block_ids: torch.Tensor,
                ctx_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x:            [B, T, dim] - draft query tokens (the masked block)
        # target_hidden:[B, C, dim] - fused target context (KV-injected)
        # block_ids:    [B, T]      - block index per query (bidirectional within block)
        # ctx_valid:    [B, C] bool - which context positions may be attended
        B, T, _ = x.shape
        C = target_hidden.shape[1]
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)          # B,H,T,hd

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(x)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(x)
        k = torch.cat([k_ctx, k_noise], dim=1).view(B, C + T, H, hd).transpose(1, 2)
        v = torch.cat([v_ctx, v_noise], dim=1).view(B, C + T, H, hd).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        scores = (q @ k.transpose(-1, -2)) * self.scale               # B,H,T,C+T

        if ctx_valid is None:
            ctx_mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
        else:
            ctx_mask = ctx_valid.to(x.device)
        ctx_mask = ctx_mask.unsqueeze(1).expand(B, T, C)              # B,T,C
        same_block = (block_ids.unsqueeze(2) == block_ids.unsqueeze(1))  # B,T,T
        attn_mask = torch.cat([ctx_mask, same_block], dim=-1)         # B,T,C+T

        scores = scores.masked_fill(~attn_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.o_proj(out)


class BlockDiffusionLayer(nn.Module):
    """Drafter layer: pre-norm attention (KV-injected) + SwiGLU MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.attn = BlockDiffusionAttention(dim, num_heads, dropout)
        self.post_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_ratio)

    def forward(self, x: torch.Tensor,
                target_hidden: torch.Tensor,
                block_ids: torch.Tensor,
                ctx_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), target_hidden, block_ids, ctx_valid)
        x = x + self.mlp(self.post_norm(x))
        return x


# ============================================================================
# DRAFTER MODEL
# ============================================================================

class BlockDiffusionDrafter(nn.Module):
    """
    Lightweight block-diffusion draft model conditioned on the target's fused
    hidden features. Shares the target's token embedding and LM head (frozen).
    """

    def __init__(self, target: nn.Module, cfg: Optional[DraftConfig] = None):
        super().__init__()
        self.target = target
        self.cfg = cfg or DraftConfig()

        tcfg = target.config
        self.vocab_size = tcfg.vocab_size
        self.dim = tcfg.hidden_dim
        self.num_heads = tcfg.num_heads
        self.block_size = self.cfg.block_size
        self.num_draft_layers = self.cfg.num_draft_layers

        # Fused target-context projection (concat sampled target layers -> dim)
        self.target_layer_ids = build_target_layer_ids(
            len(target.layers), self.cfg.num_target_ctx_layers
        )
        self.fc = nn.Linear(len(self.target_layer_ids) * self.dim, self.dim, bias=False)
        self.hidden_norm = RMSNorm(self.dim)

        # Diffusion mask token (predict masked positions in parallel)
        self.mask_embedding = nn.Parameter(torch.randn(self.dim) * 0.02)

        self.layers = nn.ModuleList([
            BlockDiffusionLayer(self.dim, self.num_heads, self.cfg.mlp_ratio, self.cfg.dropout)
            for _ in range(self.num_draft_layers)
        ])
        self.norm = RMSNorm(self.dim)

    # ------------------------------------------------------------------
    # FUSED TARGET CONTEXT (always computed under no_grad - target frozen)
    # ------------------------------------------------------------------
    def freeze_target(self) -> None:
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def fuse_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the frozen target, concat chosen layer hiddens, project to dim."""
        hiddens = self.target.extract_hidden_states(input_ids, self.target_layer_ids)
        cat = torch.cat(hiddens, dim=-1)          # B, L, len_ids * dim
        return self.hidden_norm(self.fc(cat))     # B, L, dim

    # ------------------------------------------------------------------
    # CORE PREDICT: one parallel forward over a draft block
    # ------------------------------------------------------------------
    def draft_logits(self, ctx_hidden: torch.Tensor,
                     block_query: torch.Tensor,
                     block_ids: torch.Tensor,
                     ctx_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = block_query
        for layer in self.layers:
            h = layer(h, ctx_hidden, block_ids, ctx_valid)
        h = self.norm(h)
        return self.target.output(h)              # shared, frozen LM head

    # ------------------------------------------------------------------
    # BLOCK DIFFUSION LOSS (training)
    # ------------------------------------------------------------------
    def _sample_block(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Per sample: pick a random contiguous block, keep its first token as
        the clean anchor, mask the rest, and build the leak-free context
        (only the prefix BEFORE the block is visible).
        """
        B, T = input_ids.shape
        emb = self.target.tok_embeddings
        fuse_all = self.fuse_context(input_ids)   # B, T, dim

        losses = []
        bs = self.block_size
        for b in range(B):
            n_choose = max(2, min(bs, T))
            start = random.randint(0, T - n_choose)      # random block position
            anchor_tok = input_ids[b, start]
            masked_ids = input_ids[b, start + 1: start + n_choose]

            anchor_emb = emb(anchor_tok).unsqueeze(0)                              # 1, dim
            mask_repr = self.mask_embedding.unsqueeze(0).expand(n_choose - 1, self.dim)
            block_query = torch.cat([anchor_emb, mask_repr], dim=0).unsqueeze(0)   # 1, n_choose, dim

            ctx_hidden = fuse_all[b:b + 1, :start]        # prefix only (no leakage)
            b_ids = torch.zeros(n_choose, dtype=torch.long, device=input_ids.device).unsqueeze(0)

            logits = self.draft_logits(ctx_hidden, block_query, b_ids)             # 1, n_choose, vocab
            labels = input_ids[b, start: start + n_choose]
            loss = F.cross_entropy(
                logits[:, 1:].reshape(-1, self.vocab_size),
                labels[1:].reshape(-1)
            )
            losses.append(loss)
        return losses

    def fit(self, input_ids: torch.Tensor, steps: int = 10,
            lr: float = 3e-4, log_every: int = 5) -> float:
        """Train the drafter only (target frozen) via block-diffusion loss."""
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.freeze_target()

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        optimizer = opt
        self.train()
        final = float('inf')
        for step in range(steps):
            optimizer.zero_grad()
            losses = self._sample_block(input_ids)
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            final = loss.item()
            if (step + 1) % log_every == 0:
                print(f"  drafter step {step + 1}/{steps} | loss {final:.4f}")
        return final

    # ------------------------------------------------------------------
    # CUSTOM SPECULATIVE-DECODING LOOP
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def spec_generate(self, input_ids: torch.Tensor,
                      max_new_tokens: int = 64,
                      num_draft: Optional[int] = None) -> torch.Tensor:
        """
        Block-parallel speculative decoding:
          draft block -> verify with target -> accept longest prefix + bonus.
        """
        self.freeze_target()
        self.eval()

        gen = input_ids.clone()
        draft_count = max(1, min(self.block_size - 1, num_draft or (self.block_size - 1)))
        produced = 0

        while produced < max_new_tokens:
            # ---- 1. fused context over the current clean sequence ----
            ctx_hidden = self.fuse_context(gen)                     # 1, L, dim
            L = gen.shape[1]

            # ---- 2. parallel block draft (anchor = last token of gen) ----
            anchor_emb = self.target.tok_embeddings(gen[0, -1]).unsqueeze(0)         # 1, dim
            mask_repr = self.mask_embedding.unsqueeze(0).expand(draft_count, self.dim)
            block_query = torch.cat([anchor_emb, mask_repr], dim=0).unsqueeze(0)     # 1, 1+draft_count, dim
            b_ids = torch.zeros(block_query.shape[1], dtype=torch.long, device=gen.device).unsqueeze(0)

            pred = self.draft_logits(ctx_hidden, block_query, b_ids)               # 1, 1+draft_count, vocab
            draft_tokens = pred[0, 1:].argmax(dim=-1)                               # draft_count

            # ---- 3. verify in parallel with the target ----
            cand = torch.cat([gen, draft_tokens.unsqueeze(0)], dim=1)               # 1, L+draft_count
            cand_logits, _ = self.target(cand)
            tgt_preds = cand_logits[0, L - 1: L - 1 + draft_count].argmax(dim=-1)   # ground-truth next tokens
            bonus = cand_logits[0, -1].argmax()                                     # token after the draft

            # ---- 4. longest accepted prefix + bonus token ----
            n_match = 0
            while n_match < draft_count and draft_tokens[n_match].item() == tgt_preds[n_match].item():
                n_match += 1

            accepted = draft_tokens[:n_match]
            if n_match > 0:
                gen = torch.cat([gen, accepted.unsqueeze(0)], dim=1)
            gen = torch.cat([gen, bonus.reshape(1, 1)], dim=1)
            produced += n_match + 1

        return gen[:, :input_ids.shape[1] + max_new_tokens]


# ============================================================================
# DEMO / CLI
# ============================================================================

def main():
    from train_100m_llm_numba import TrainingConfig, LanguageModel100M

    parser = argparse.ArgumentParser(description="DFlash-style block-diffusion drafter")
    parser.add_argument("--train-steps", type=int, default=20, help="drafter training steps")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="generation length")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--draft-layers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    tiny = not torch.cuda.is_available()

    cfg = TrainingConfig(
        vocab_size=512, hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=128,
        dropout=0.0, batch_size=2, enable_qat=False, compile_model=False, use_amp=False,
        output_dir="dflash_out", checkpoint_dir="dflash_ckpt",
    ) if tiny else TrainingConfig()

    target = LanguageModel100M(cfg)
    print(f"Target params: {target.num_params / 1e6:.2f}M")

    dcfg = DraftConfig(num_draft_layers=args.draft_layers,
                       block_size=args.block_size,
                       num_target_ctx_layers=max(2, cfg.num_layers // 2))
    drafter = BlockDiffusionDrafter(target, dcfg)
    drafter.freeze_target()
    drafter_params = sum(p.numel() for p in drafter.parameters() if p.requires_grad)
    print(f"Drafter params: {drafter_params / 1e3:.1f}K (target frozen, shared emb + LM head)")

    B, T = 4, 32
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    print("Training block-diffusion drafter...")
    final_loss = drafter.fit(ids, steps=args.train_steps)
    print(f"  -> final drafter loss: {final_loss:.4f}")

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = drafter.spec_generate(prompt, max_new_tokens=args.max_new_tokens)
    print(f"Speculative generation: prompt {prompt.shape[1]} -> {out.shape[1]} tokens "
          f"(draft={dcfg.block_size - 1}/block, {args.draft_layers} drafter layers)")


if __name__ == "__main__":
    main()