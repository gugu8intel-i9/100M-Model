# 100M-Model
LLM Optimizer Benchmark and Training Scripts for ~100M Parameter Transformers

## Overview
This repository contains comprehensive scripts for training and benchmarking 100M parameter language models, with comparison across 10 different optimizers.

## Current Results
### Optimizer Performance Ranking (Lower Loss = Better)

| Rank | Optimizer | Final Loss | Status |
|------|-----------|------------|--------|
| 1 | **Lion** | **8.2268** | ✅ Converged (BEST) |
| 2 | AdamW | 8.2643 | ✅ Converged |
| 3 | Adafactor | 8.2693 | ✅ Converged |
| 4 | Prodigy | 8.3102 | ✅ Converged (parameter-free) |
| 5 | Muon | 8.3671 | ✅ Converged |
| 6 | RAdam | 8.4416 | ✅ Converged |
| - | Sophia | NaN/Inf | ❌ Diverged |
| - | Adan | NaN/Inf | ❌ Diverged |
| - | SF-AdamW | NaN/Inf | ❌ Diverged |
| - | D-Adam | NaN/Inf | ❌ Diverged |

### Key Findings
- **Lion** achieves the best performance (8.2268), closely followed by AdamW (8.2643) - only ~0.5% difference
- **Prodigy** demonstrates parameter-free optimization is viable with minimal performance tradeoff (8.3102)
- **Muon** (SOTA Dec 2024) performs well at this scale, though theoretical 2x speedups may manifest at 100B+ parameter scales
- Four optimizers (Sophia, Adan, SF-AdamW, D-Adam) failed to converge, likely requiring different hyperparameter initialization or longer training runs

### Best Hyperparameters (Bayesian-optimized)
- **Learning rate:** 1.75e-4 (range: 1e-4 to 3e-4)
- **Weight decay:** 0.149 (range: 0.1 to 0.2)
- **Beta values:** (0.9, 0.95) for AdamW/Lion
- **Scheduler:** Cosine annealing with 500-1000 step warmup

## Model Architecture
- ~100M parameter transformer (12 layers × 768 dim × 12 heads × 4x MLP)
- Mixed precision (FP16) training
- Causal language modeling objective
- Synthetic dataset for rapid iteration

## Scripts Included
- `scripts/llm_100m_gpu_numba_qat.py` - Main training script with Numba JIT acceleration and QAT (INT8/INT4/ternary)
- `scripts/llm_fast_benchmark_v31.py` - Fast benchmark v3.1 (5 trials/optimizer)
- `scripts/generate_benchmark_report.py` - Professional PDF report generator
- `scripts/train_100m_llm_numba.py` - 100M LLM training script
- `scripts/llm_modal_deploy.py` - Modal cloud GPU deployment

## Results Data
- `download/fast_benchmark_v31_1787004215.json` - Benchmark results JSON
- `download/best_llm_hyperparams.json` - Best hyperparameters from search

## Contributors
- **Kunal** - LLM 100M parameter optimizer benchmark and training scripts
  - Conducted comprehensive optimizer benchmark across 10 optimizers
  - Trained model with Numba JIT acceleration and QAT support
  - Generated benchmark reports and visualizations
  - Authored training scripts with Bayesian-optimized hyperparameters

## Changelog
### v3.3 (Current)
- **Dropped** the DeepSeek-V3 MTP blocks
- Added **DFlash-style block-diffusion drafter** (`scripts/dflash_drafter.py`, after arXiv 2602.06036)
  - KV-injection attention: fused target-context features injected into K/V of every drafter layer
  - Block-diffusion training: random anchor seeds each block; masked positions predicted in parallel; bidirectional attention inside a block only (no cross-block leakage)
  - `spec_generate()`: custom speculative-decoding loop — draft block, verify with target, accept longest prefix + bonus token
  - Drafter shares the frozen target's token embedding + LM head (stays small); adds `extract_hidden_states()` to the target for fused context

### v3.2
- Fixed import crash when Numba is unavailable (no-op `jit`/`prange` fallbacks)

### v3.1
- Fast benchmark with 5 trials per optimizer
- Added Lion, Sophia, Adan, RAdam, Muon, Prodigy, SF-AdamW, D-Adam
- Bayesian-optimized hyperparameters (LR=1.75e-4, WD=0.149, AdamW)
- Numba JIT-accelerated functions for attention, loss, gradients
- QAT support: INT8, INT4, Ternary (BitNet b1.58 style)
- Modal cloud GPU deployment ready

### v3.0
- Initial release with full optimizer comparison
- Comprehensive benchmark report generation
- Multi-optimizer training framework

## Repository
- Local: `/home/sanjay/Downloads/Hyperpram Optimaization & Optimizer Result/`
- Git: Initialized with all scripts and results
- GitHub: https://github.com/gugu8intel-i9/100M-Model (reference project)

## Usage
```bash
# Run fast benchmark
python scripts/llm_fast_benchmark_v31.py

# Train with specific optimizer
python scripts/llm_100m_gpu_numba_qat.py --optimizer lion --lr 1.75e-4 --wd 0.149

# Train target model (no MTP - plain transformer + QAT)
python scripts/train_100m_llm_numba.py --mode train

# Train DFlash-style block-diffusion drafter + speculative generate
python scripts/dflash_drafter.py --train-steps 20 --max-new-tokens 64
```