LLM Optimizer Benchmark Results - 100M Parameter Model
========================================================

Overview
--------
This repository contains the complete benchmark suite for evaluating optimizers
on a ~100M parameter language model training task. Results were obtained using
a scaled 4.2M parameter model for rapid iteration, with findings extrapolated
to the 100M target scale.

Key Results
-----------
Final Loss (lower is better) | Optimizer | Status
------------------------------|-----------|--------
8.2268 | Lion | ✅ Converged (BEST)
8.2643 | AdamW | ✅ Converged
8.2693 | Adafactor | ✅ Converged
8.3102 | Prodigy | ✅ Converged (parameter-free)
8.3671 | Muon | ✅ Converged
8.4416 | RAdam | ✅ Converged
Failed (NaN/Inf) | Sophia | ❌ Diverged
Failed (NaN/Inf) | Adan | ❌ Diverged
Failed (NaN/Inf) | SF-AdamW | ❌ Diverged
Failed (NaN/Inf) | D-Adam | ❌ Diverged

Performance Ranking
-----------------
1. Lion (8.2268) - Sign-based momentum, implicit regularization
2. AdamW (8.2643) - Decoupled weight decay, warm cosine LR
3. Adafactor (8.2693) - Factorized moments, memory-efficient
4. Prodigy (8.3102) - Parameter-free, auto-LR from gradient stats
5. Muon (8.3671) - SVD-based Newton steps, Dec 2024 SOTA
6. RAdam (8.4416) - Rectified Adam, variance correction

Hyperparameters (Bayesian-optimized)
-------------------------------------
- Learning rate: 1.75e-4 (range: 1e-4 to 3e-4)
- Weight decay: 0.149 (range: 0.1 to 0.2)
- Beta values: (0.9, 0.95) for AdamW/Lion
- Scheduler: Cosine annealing with 500-1000 step warmup

Key Findings
------------
1. Classical optimizers (AdamW, Adafactor) remain highly competitive despite novel methods
2. Parameter-free methods (Prodigy) are viable with minimal performance tradeoff
3. Scale matters - newer optimizers (Muon, Sophia) may perform better at 100B+ scale
4. Numerical stability varies widely between optimizers
5. Hyperparameter search remains essential even for "parameter-free" methods

Files Included
--------------
- scripts/llm_100m_gpu_numba_qat.py - Main training script with Numba QAT
- scripts/llm_fast_benchmark_v31.py - Fast benchmark v3.1
- scripts/generate_benchmark_report.py - Professional PDF report generator
- download/fast_benchmark_v31_1787004215.json - Benchmark results JSON
- download/best_llm_hyperparams.json - Best hyperparameters from search
- CONTRIBUTORS - Contributor credits and detailed results summary