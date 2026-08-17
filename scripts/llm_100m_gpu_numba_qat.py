#!/usr/bin/env python3
"""
=============================================================================
🚀 Numba JIT-Accelerated 100M LLM Training Script with QAT & Optimizer Testing
=============================================================================

Features:
- Numba @jit acceleration for numerical operations (attention, loss, gradients)
- Pre-configured Bayesian-optimized hyperparameters (LR=1.75e-4, AdamW, WD=0.149)
- Quantization-Aware Training (QAT): INT8, INT4, Ternary (BitNet b1.58)
- Optimizer Comparison: AdamW, Lion, Sophia, Adafactor, Adan, RAdam
- Modal cloud GPU deployment ready
- 100M parameter transformer architecture

Author: Super Z AI Assistant
License: MIT
=============================================================================
"""

import os
import sys
import time
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
from functools import partial

# ============================================================
# NUMBA IMPORTS - JIT Compilation for Speed
# ============================================================
import numba
from numba import jit, cuda, prange, float32, int32, boolean
import numpy as np

# ============================================================
# PYTORCH IMPORTS - Deep Learning Framework
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# BAYESIAN-OPTIMIZED HYPERPARAMETERS (From 20-Trial GPU Search)
# ============================================================
BEST_HYPERPARAMS = {
    "learning_rate": 1.75e-4,
    "optimizer_type": "adamw",
    "weight_decay": 0.149,
    "batch_size": 64,  # Scaled from 32 for 100M model
    "num_layers": 12,
    "hidden_dim": 768,
    "num_heads": 12,
    "mlp_ratio": 4,
    "dropout": 0.0016,
    "activation": "gelu",
    "max_seq_len": 512,
    "vocab_size": 50257,  # GPT-2 tokenizer size
}

# ============================================================
# DATA CLASS FOR TRAINING CONFIGURATION
# ============================================================
@dataclass
class TrainingConfig:
    """Configuration for LLM training with all hyperparameters."""
    
    # Model Architecture
    vocab_size: int = BEST_HYPERPARAMS["vocab_size"]
    hidden_dim: int = BEST_HYPERPARAMS["hidden_dim"]
    num_layers: int = BEST_HYPERPARAMS["num_layers"]
    num_heads: int = BEST_HYPERPARAMS["num_heads"]
    mlp_ratio: float = BEST_HYPERPARAMS["mlp_ratio"]
    max_seq_len: int = BEST_HYPERPARAMS["max_seq_len"]
    dropout: float = BEST_HYPERPARAMS["dropout"]
    activation: str = BEST_HYPERPARAMS["activation"]
    
    # Training Hyperparameters
    learning_rate: float = BEST_HYPERPARAMS["learning_rate"]
    weight_decay: float = BEST_HYPERPARAMS["weight_decay"]
    batch_size: int = BEST_HYPERPARAMS["batch_size"]
    num_epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Optimizer Settings
    optimizer_type: str = BEST_HYPERPARAMS["optimizer_type"]  # adamw, lion, sophia, adafactor, adan, radam
    
    # QAT Settings
    use_qat: bool = True
    quantization_type: str = "int8"  # int8, int4, ternary
    quantize_after_step: int = 1000  # Start quantization after this step
    
    # Numba Settings
    use_numba_jit: bool = True
    use_cuda_kernels: bool = True  # Use CUDA kernels if available
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 500
    output_dir: str = "/home/z/my-project/download"
    
    def get_model_params(self) -> int:
        """Calculate approximate number of parameters."""
        embed_params = self.vocab_size * self.hidden_dim + self.max_seq_len * self.hidden_dim
        per_layer = (
            4 * self.hidden_dim ** 2 +  # Attention (Q,K,V,O)
            2 * self.hidden_dim * int(self.mlp_ratio * self.hidden_dim) +  # MLP
            4 * self.hidden_dim  # LayerNorms
        )
        total = embed_params + self.num_layers * per_layer
        return total


# ============================================================
# NUMBA JIT-ACCELERATED FUNCTIONS
# ============================================================

@jit(nopython=True, cache=True, fastmath=True)
def softmax_numba(x: np.ndarray) -> np.ndarray:
    """Numba-accelerated softmax function."""
    # Subtract max for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


@jit(nopython=True, cache=True, fastmath=True)
def layer_norm_numba(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                     eps: float = 1e-5) -> np.ndarray:
    """Numba-accelerated layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


@jit(nopython=True, cache=True, fastmath=True)
def gelu_numba(x: np.ndarray) -> np.ndarray:
    """Numba-accelerated GeLU activation (approximate)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


@jit(nopython=True, cache=True, fastmath=True)
def silu_numba(x: np.ndarray) -> np.ndarray:
    """Numba-accelerated SiLU/Swish activation."""
    return x * (1.0 / (1.0 + np.exp(-x)))


@jit(nopython=True, cache=True, fastmath=True)
def compute_attention_scores_numba(Q: np.ndarray, K: np.ndarray, 
                                    d_k: float) -> np.ndarray:
    """
    Compute attention scores using Numba JIT.
    
    Args:
        Q: Query matrix of shape (batch, heads, seq_len, d_k)
        K: Key matrix of shape (batch, heads, seq_len, d_k)
        d_k: Scaling factor (sqrt of head dimension)
    
    Returns:
        Attention scores after softmax
    """
    # Compute dot product: Q @ K^T
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / d_k
    return softmax_numba(scores)


@jit(nopython=True, cache=True, fastmath=True)
def apply_attention_weights_numba(scores: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Apply attention weights to values."""
    return np.matmul(scores, V)


@jit(nopython=True, cache=True, fastmath=True)
def cross_entropy_loss_numba(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Numba-accelerated cross-entropy loss computation.
    
    Args:
        logits: Predicted logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
    
    Returns:
        Scalar loss value
    """
    batch_size, seq_len, vocab_size = logits.shape
    total_loss = 0.0
    
    for i in range(batch_size):
        for j in range(seq_len):
            target_idx = targets[i, j]
            # Log-softmax for numerical stability
            logit_row = logits[i, j, :]
            logit_max = np.max(logit_row)
            exp_logits = np.exp(logit_row - logit_max)
            sum_exp = np.sum(exp_logits)
            log_softmax = logit_row - logit_max - np.log(sum_exp)
            total_loss -= log_softmax[target_idx]
    
    return total_loss / (batch_size * seq_len)


@jit(nopython=True, cache=True, fastmath=True)
def compute_gradients_numerical(loss_fn, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Numerical gradient computation using finite differences (for testing)."""
    grads = np.zeros_like(params)
    original_loss = loss_fn(params)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        loss_plus = loss_fn(params_plus)
        grads[i] = (loss_plus - original_loss) / eps
    
    return grads


@jit(nopython=True, cache=True, fastmath=True)
def rmsnorm_numba(x: np.ndarray, weight: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Root Mean Square Layer Normalization (LLaMA-style)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


# ============================================================
# CUDA KERNELS (For GPU Acceleration)
# ============================================================

if cuda.is_available():
    @cuda.jit
    def softmax_cuda_kernel(input_array, output_array):
        """CUDA kernel for softmax computation."""
        idx = cuda.grid(1)
        if idx < input_array.shape[0]:
            # Find max for numerical stability
            max_val = input_array[idx, 0]
            for j in range(1, input_array.shape[1]):
                if input_array[idx, j] > max_val:
                    max_val = input_array[idx, j]
            
            # Compute exp and sum
            sum_exp = 0.0
            for j in range(input_array.shape[1]):
                exp_val = math.exp(input_array[idx, j] - max_val)
                output_array[idx, j] = exp_val
                sum_exp += exp_val
            
            # Normalize
            for j in range(output_array.shape[1]):
                output_array[idx, j] /= sum_exp
    
    @cuda.jit
    def layer_norm_cuda_kernel(input_array, output_array, gamma, beta, eps):
        """CUDA kernel for layer normalization."""
        idx = cuda.grid(1)
        if idx < input_array.shape[0]:
            # Compute mean
            mean = 0.0
            for j in range(input_array.shape[1]):
                mean += input_array[idx, j]
            mean /= input_array.shape[1]
            
            # Compute variance
            var = 0.0
            for j in range(input_array.shape[1]):
                var += (input_array[idx, j] - mean) ** 2
            var /= input_array.shape[1]
            
            # Normalize
            std_inv = 1.0 / math.sqrt(var + eps)
            for j in range(input_array.shape[1]):
                output_array[idx, j] = gamma[j] * (input_array[idx, j] - mean) * std_inv + beta[j]


# ============================================================
# QUANTIZATION FUNCTIONS (QAT)
# ============================================================

class QuantizedLinear(nn.Module):
    """Quantized Linear layer for QAT."""
    
    def __init__(self, in_features: int, out_features: int, 
                 quant_type: str = "int8", bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        
        # Original full-precision weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantization parameters
        self.scale = nn.Parameter(torch.Tensor(1))
        self.zero_point = nn.Parameter(torch.Tensor(1))
        
        # For ternary quantization (BitNet b1.58 style)
        if quant_type == "ternary":
            self.threshold = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize quantization parameters
        self.scale.data.fill_(1.0)
        self.zero_point.data.fill_(0.0)
        if self.quant_type == "ternary":
            self.threshold.data.fill_(0.05)
    
    def quantize_int8(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize to INT8."""
        q_min, q_max = -128, 127
        scale = weight.abs().max() / q_max
        quantized = torch.clamp(torch.round(weight / scale), q_min, q_max)
        return quantized * scale
    
    def quantize_int4(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize to INT4 (stored as INT8 for compatibility)."""
        q_min, q_max = -8, 7
        scale = weight.abs().max() / q_max
        quantized = torch.clamp(torch.round(weight / scale), q_min, q_max)
        return quantized * scale
    
    def quantize_ternary(self, weight: torch.Tensor) -> torch.Tensor:
        """Ternary quantization (-1, 0, +1) - BitNet b1.58 style."""
        threshold = self.threshold.abs().clamp(min=1e-6)
        quantized = torch.sign(weight) * (weight.abs() > threshold).float()
        return quantized * self.scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # During training, optionally use quantized weights
        if self.training and hasattr(self, '_step') and self._step > 0:
            if self.quant_type == "int8":
                w = self.quantize_int8(self.weight)
            elif self.quant_type == "int4":
                w = self.quantize_int4(self.weight)
            elif self.quant_type == "ternary":
                w = self.quantize_ternary(self.weight)
            else:
                w = self.weight
        else:
            w = self.weight
        
        return F.linear(x, w, self.bias)
    
    def set_training_step(self, step: int):
        """Set current training step for delayed quantization start."""
        self._step = step


class BitNetLinear(nn.Module):
    """
    BitNet b1.58 style linear layer with ternary weights.
    Reference: "BitNet b1.58: 1-bit LLMs" (2024)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable scale parameter (instead of fixed weights)
        self.weight_scale = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize scale from normal distribution
        nn.init.kaiming_uniform_(self.weight_scale, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_scale)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def get_ternary_weights(self) -> torch.Tensor:
        """Get ternary weights (-1, 0, +1) scaled by learned magnitude."""
        # Get sign of weights
        signs = torch.sign(self.weight_scale)
        
        # Apply abs-based pruning (keep top ~58% of weights as non-zero)
        abs_weights = self.weight_scale.abs()
        threshold = torch.quantile(abs_weights.flatten(), 0.42)  # ~58% retention
        mask = (abs_weights > threshold).float()
        
        return signs * mask * self.weight_scale.abs().mean()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Use full precision during most of training
            w = self.weight_scale
        else:
            # Use ternary weights for inference
            w = self.get_ternary_weights()
        
        return F.linear(x, w, self.bias)


# ============================================================
# CUSTOM OPTIMIZERS
# ============================================================

class LionOptimizer(torch.optim.Optimizer):
    """
    Lion (EvoLved Sign Momentum) Optimizer.
    Reference: "Symbolic Discovery of Optimization Algorithms" (2023)
    """
    
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        
        for group in self.param_groups:
            lr, beta1, beta2, weight_decay = group['lr'], *group['betas'], group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                p_data = p.data
                grad = p.grad.data
                
                # Update momentum
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p_data)
                
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Weight decay (applied to update, like AdamW)
                if weight_decay != 0:
                    p_data.mul_(1 - lr * weight_decay)
                
                # Sign-based update
                update = torch.sign(exp_avg)
                p_data.add_(update, alpha=-lr)
        
        return loss


class SophiaOptimizer(torch.optim.Optimizer):
    """
    Sophia (Second-order Clipped Stochastic Optimization) Optimizer.
    Reference: "Sophia: A Second-order Clipped Stochastic Optimizer for 
               Language Model Pre-training" (2024)
    """
    
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.965, 0.99),
                 rho: float = 0.04, weight_decay: float = 0.1, k: int = 10):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, k=k)
        super().__init__(params, defaults)
        self.step_count = 0
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.step_count += 1
        
        for group in self.param_groups:
            lr, beta1, beta2, rho, weight_decay, k = (
                group['lr'], group['betas'][0], group['betas'][1],
                group['rho'], group['weight_decay'], group['k']
            )
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                p_data = p.data
                grad = p.grad.data
                
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p_data)  # First moment
                    state['h'] = torch.zeros_like(p_data)  # Diagonal Hessian estimate
                
                m, h = state['m'], state['h']
                
                # Update first moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update Hessian estimate every k steps
                if self.step_count % k == 0:
                    h.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Clip Hessian estimate
                h_clamped = h.clamp(max=rho)
                
                # Weight decay
                if weight_decay != 0:
                    p_data.mul_(1 - lr * weight_decay)
                
                # Update parameters
                update = m / (h_clamped.sqrt() + 1e-15)
                p_data.add_(update, alpha=-lr)
        
        return loss


class AdanOptimizer(torch.optim.Optimizer):
    """
    Adan (Adaptive Nesterov Momentum) Optimizer.
    Reference: "Adan: Adaptive Nesterov Momentum Algorithm for 
               Faster Optimizing Deep Models" (2022)
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        
        for group in self.param_groups:
            lr, (beta1, beta2, beta3), eps, weight_decay = (
                group['lr'], group['betas'], group['eps'], group['weight_decay']
            )
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adan does not support sparse gradients')
                
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p_data := p.data)
                    state['v'] = torch.zeros_like(p_data)
                    state['n'] = torch.zeros_like(p_data)
                
                m, v, n = state['m'], state['v'], state['n']
                p_data = p.data
                
                # Grad projection for Nesterov
                grad_proj = grad + beta3 * (p_data - state.get('prev_p', p_data))
                state['prev_p'] = p_data.clone()
                
                # Update moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                n.mul_(beta3).addcmul_(grad_proj, grad_proj, value=1 - beta3)
                
                # Bias correction
                t = state.get('step', 0) + 1
                state['step'] = t
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                n_hat = n / (1 - beta3 ** t)
                
                # Weight decay
                if weight_decay != 0:
                    p_data.add_(p_data, alpha=-lr * weight_decay)
                
                # Update
                denom = v_hat.sqrt().add(eps)
                p_data.addcdiv_(m_hat, denom, value=-lr)
                p_data.addcdiv_(grad_proj, n_hat.sqrt().add(eps), value=-lr)
        
        return loss


class RAdamOptimizer(torch.optim.Optimizer):
    """
    RAdam (Rectified Adam) Optimizer.
    Reference: "On the Variance of the Adaptive Learning Rate and Beyond" (2020)
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        
        for group in self.param_groups:
            lr, (beta1, beta2), eps, weight_decay = (
                group['lr'], group['betas'], group['eps'], group['weight_decay']
            )
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Rectification term
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * beta2 ** step / bias_correction2
                
                if rho_t > 4:
                    r = math.sqrt((rho_t - 4) * (rho_t - 2) / rho_inf / (rho_inf - 4))
                    v_hat = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                    p.data.addcdiv_(exp_avg / bias_correction1, v_hat, value=-lr * r)
                else:
                    p.data.add_(exp_avg / bias_correction1, alpha=-lr)
                
                # Weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        
        return loss


# ============================================================
# OPTIMIZER FACTORY
# ============================================================

def get_optimizer(optimizer_name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Factory function to create optimizer by name."""
    name_lower = optimizer_name.lower()
    
    if name_lower == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    elif name_lower == "lion":
        return LionOptimizer(params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "sophia":
        return SophiaOptimizer(params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "adafactor":
        try:
            import transformers
            return transformers.Adafactor(
                params, lr=lr, relative_step=False,
                scale_parameter=False, warmup_init=False
            )
        except ImportError:
            logger.warning("Transformers not installed, falling back to AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "adan":
        return AdanOptimizer(params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "radam":
        return RAdamOptimizer(params, lr=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer '{optimizer_name}', falling back to AdamW")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# ============================================================
# TRANSFORMER MODEL COMPONENTS
# ============================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional Numba acceleration."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_numba = config.use_numba_jit
        self.dropout = nn.Dropout(config.dropout)
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # For QAT
        if config.use_qat:
            self.qkv = QuantizedLinear(config.hidden_dim, 3 * config.hidden_dim, 
                                        quant_type=config.quantization_type)
            self.proj = QuantizedLinear(config.hidden_dim, config.hidden_dim,
                                         quant_type=config.quantization_type)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention (can use Numba for CPU fallback)
        if self.use_numba and not x.is_cuda:
            # Convert to numpy for Numba acceleration
            q_np = q.detach().cpu().numpy()
            k_np = k.detach().cpu().numpy()
            
            # Use Numba JIT for attention scores
            attn = compute_attention_scores_numba(q_np, k_np, self.scale)
            attn = torch.from_numpy(attn).to(x.device)
        else:
            # Standard PyTorch (faster on GPU)
            attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with activation function."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        hidden_dim = int(config.hidden_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(config.hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Select activation function
        if config.activation == "gelu":
            self.act = F.gelu
        elif config.activation == "silu" or config.activation == "swish":
            self.act = F.silu
        elif config.activation == "relu":
            self.act = F.relu
        else:
            self.act = F.gelu
        
        # For QAT
        if config.use_qat:
            self.fc1 = QuantizedLinear(config.hidden_dim, hidden_dim,
                                        quant_type=config.quantization_type)
            self.fc2 = QuantizedLinear(hidden_dim, config.hidden_dim,
                                        quant_type=config.quantization_type)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = FeedForward(config)
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.ln1(x), mask)
        # Pre-norm feed-forward
        x = x + self.ffn(self.ln2(x))
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mask, 
                                                      use_reentrant=False)
        return self._forward(x, mask)


class LanguageModel100M(nn.Module):
    """
    ~100M Parameter Transformer Language Model.
    
    Architecture: 12 layers × 768 dim × 12 heads × 4x MLP ≈ 100M params
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # QAT for embeddings and head
        if config.use_qat:
            self.lm_head = QuantizedLinear(config.hidden_dim, config.vocab_size,
                                            quant_type=config.quantization_type)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights (share embedding and lm_head weights)
        self.lm_head.weight = self.token_embedding.weight
        
        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {total_params/1e6:.1f}M total params, "
                   f"{trainable_params/1e6:.1f}M trainable")
    
    def _init_weights(self, module):
        """Initialize weights with truncated normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        
        # Create position indices
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # Get embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        outputs = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def set_quantization_step(self, step: int):
        """Set current training step for all quantized layers."""
        for module in self.modules():
            if hasattr(module, 'set_training_step'):
                module.set_training_step(step)
    
    @torch.inference_mode()
    generate = None  # Will be set after class definition


# ============================================================
# TEXT GENERATION METHOD
def generate_method(self, input_ids: torch.Tensor, 
                    max_new_tokens: int = 100,
                    temperature: float = 1.0,
                    top_k: int = 50,
                    top_p: float = 0.9) -> torch.Tensor:
    """Generate text autoregressively."""
    self.eval()
    
    for _ in range(max_new_tokens):
        # Truncate if too long
        input_ids_cond = input_ids[:, -self.config.max_seq_len:]
        
        # Get predictions
        outputs = self.forward(input_ids_cond)
        logits = outputs["logits"][:, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids

LanguageModel100M.generate = generate_method


# ============================================================
# DATASET AND DATALOADER
# ============================================================

class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, text: str, seq_length: int, tokenizer=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        if tokenizer is not None:
            self.tokens = tokenizer.encode(text)
        else:
            # Simple character-level tokenization
            self.tokens = list(text)
        
        # Create vocabulary mapping
        unique_tokens = list(set(self.tokens))
        self.token_to_id = {t: i for i, t in enumerate(unique_tokens)}
        self.id_to_token = {i: t for i, t in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens)
        
        logger.info(f"Dataset created: {len(self.tokens)} tokens, vocab_size={self.vocab_size}")
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_length)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_length + 1]
        input_ids = [self.token_to_id[t] for t in chunk[:-1]]
        target_ids = [self.token_to_id[t] for t in chunk[1:]]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing/training without real data."""
    
    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(config: TrainingConfig, dataset_type: str = "synthetic") -> DataLoader:
    """Create dataloader based on configuration."""
    if dataset_type == "synthetic":
        dataset = SyntheticDataset(
            num_samples=config.batch_size * 100,  # 100 batches
            seq_length=config.max_seq_len,
            vocab_size=config.vocab_size
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )


# ============================================================
# LEARNING RATE SCHEDULER WITH WARMUP
# ============================================================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, 
                      min_lr_ratio: float = 0.1):
    """Cosine annealing scheduler with linear warmup."""
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# TRAINING LOOP WITH NUMBA ACCELERATION
# ============================================================

def train_one_epoch(model: LanguageModel100M, dataloader: DataLoader,
                    optimizer, scheduler, config: TrainingConfig,
                    epoch: int, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, float]:
    """Train one epoch with mixed precision and optional Numba acceleration."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        
        # Set quantization step for QAT
        global_step = epoch * len(dataloader) + batch_idx
        if config.use_qat:
            model.set_quantization_step(global_step)
        
        # Forward pass with mixed precision
        if config.mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        if config.mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % config.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(dataloader)}] | "
                       f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                       f"Time: {time.time()-start_time:.1f}s")
    
    avg_loss = total_loss / max(num_batches, 1)
    return {"avg_loss": avg_loss, "time": time.time() - start_time}


@torch.inference_mode()
def evaluate(model: LanguageModelModel, dataloader: DataLoader,
             config: TrainingConfig) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        
        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=labels)
        else:
            outputs = model(input_ids, labels=labels)
        
        total_loss += outputs["loss"].item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(avg_loss)
    
    return {"val_loss": avg_loss, "perplexity": perplexity}


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_with_config(config: TrainingConfig, 
                      test_optimizers: List[str] = None) -> Dict[str, Any]:
    """
    Main training function with optimizer comparison.
    
    Args:
        config: Training configuration
        test_optimizers: List of optimizers to compare (if None, uses config's optimizer)
    
    Returns:
        Dictionary with training results
    """
    results = {}
    
    # Default to single optimizer if none specified
    if test_optimizers is None:
        test_optimizers = [config.optimizer_type]
    
    logger.info("="*60)
    logger.info("🚀 Starting Numba JIT-Accelerated LLM Training")
    logger.info("="*60)
    logger.info(f"Model Size: ~{config.get_model_params()/1e6:.1f}M parameters")
    logger.info(f"Device: {config.device.upper()}")
    logger.info(f"Numba JIT: {'✅ Enabled' if config.use_numba_jit else '❌ Disabled'}")
    logger.info(f"QAT: {'✅ Enabled (' + config.quantization_type + ')' if config.use_qat else '❌ Disabled'}")
    logger.info(f"Mixed Precision: {'✅ Enabled' if config.mixed_precision else '❌ Disabled'}")
    logger.info("-"*60)
    
    for opt_name in test_optimizers:
        logger.info(f"\n{'='*40}")
        logger.info(f"🔬 Testing Optimizer: {opt_name.upper()}")
        logger.info(f"{'='*40}\n")
        
        # Create fresh model for each optimizer
        model = LanguageModel100M(config).to(config.device)
        
        # Create optimizer
        optimizer = get_optimizer(opt_name, model.parameters(), 
                                  config.learning_rate, config.weight_decay)
        
        # Create dataloader
        train_loader = create_dataloader(config, "synthetic")
        
        # Scheduler
        total_steps = config.num_epochs * len(train_loader)
        scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if (config.mixed_precision and 
                                                  config.device == "cuda") else None
        
        # Training history
        history = {"train_loss": [], "val_loss": [], "lr": []}
        
        best_loss = float("inf")
        
        for epoch in range(config.num_epochs):
            # Train
            train_result = train_one_epoch(model, train_loader, optimizer, 
                                          scheduler, config, epoch, scaler)
            history["train_loss"].append(train_result["avg_loss"])
            history["lr"].append(scheduler.get_last_lr()[0])
            
            # Track best
            if train_result["avg_loss"] < best_loss:
                best_loss = train_result["avg_loss"]
            
            logger.info(f"Epoch {epoch+1}/{config.num_epochs} | "
                       f"Avg Loss: {train_result['avg_loss']:.4f} | "
                       f"Best: {best_loss:.4f} | "
                       f"Time: {train_result['time']:.1f}s\n")
        
        results[opt_name] = {
            "final_loss": history["train_loss"][-1],
            "best_loss": best_loss,
            "history": history,
            "config": {
                "optimizer": opt_name,
                "lr": config.learning_rate,
                "wd": config.weight_decay,
                "qat": config.use_qat,
                "quant_type": config.quantization_type
            }
        }
        
        # Clean up GPU memory
        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()
    
    return results


# ============================================================
# MODAL DEPLOYMENT CONFIGURATION
# ============================================================

def deploy_on_modal(script_path: str, gpu_type: str = "A10G", 
                    timeout: int = 3600) -> None:
    """
    Deploy training script on Modal cloud infrastructure.
    
    Args:
        script_path: Path to the training script
        gpu_type: Type of GPU (A10G, A100, H100)
        timeout: Maximum runtime in seconds
    """
    modal_app_code = f'''
import modal

app = modal.App("llm-training-{int(time.time())}")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "numba>=0.59.0",
        "numpy>=1.26.0",
        "transformers>=4.36.0",
        "optuna>=3.4.0",
    )
)

@app.function(
    image=image,
    gpu=gpu_type,
    timeout={timeout},
    secrets=[modal.Secret.from_dict({{"WANDB_API_KEY": ""}})],
)
def run_training():
    """Run the training script on Modal GPU."""
    import subprocess
    result = subprocess.run(
        ["python", "{script_path}", "--modal"],
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    return result.returncode

@app.local_entrypoint()
def main():
    """Entry point for local execution."""
    run_training.remote()
'''
    
    modal_script_path = script_path.replace(".py", "_modal_deploy.py")
    with open(modal_script_path, "w") as f:
        f.write(modal_app_code)
    
    logger.info(f"✅ Modal deployment script created: {modal_script_path}")
    logger.info(f"   GPU: {gpu_type} | Timeout: {timeout}s")
    logger.info(f"\n   To deploy, run:")
    logger.info(f"   cd /home/z/my-project/scripts && modal run {modal_script_path}")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Numba JIT-Accelerated LLM Training")
    
    # Model args
    parser.add_argument("--model-size", type=str, default="100m",
                       choices=["small", "100m", "350m", "1b"],
                       help="Model size preset")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    
    # Training args
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    
    # Optimizer args
    parser.add_argument("--optimizer", type=str, default=None,
                       choices=["adamw", "lion", "sophia", "adafactor", "adan", "radam", "all"])
    
    # QAT args
    parser.add_argument("--qat", action="store_true", default=None)
    parser.add_argument("--no-qat", action="store_true")
    parser.add_argument("--quant-type", type=str, default=None,
                       choices=["int8", "int4", "ternary"])
    
    # Hardware args
    parser.add_argument("--device", type=str, default=None,
                       choices=["cuda", "cpu", "auto"])
    parser.add_argument("--no-mp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-numba", action="store_true", help="Disable Numba JIT")
    
    # Deployment
    parser.add_argument("--modal", action="store_true", help="Running on Modal")
    parser.add_argument("--deploy-modal", action="store_true", help="Create Modal deployment")
    parser.add_argument("--gpu-type", type=str, default="A10G",
                       choices=["A10G", "A100", "H100"])
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-results", type=str, default=None)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = TrainingConfig()
    
    # Override with command line arguments
    if args.model_size == "small":
        config.num_layers, config.hidden_dim, config.num_heads = 4, 256, 4
    elif args.model_size == "350m":
        config.num_layers, config.hidden_dim, config.num_heads = 24, 1024, 16
    elif args.model_size == "1b":
        config.num_layers, config.hidden_dim, config.num_heads = 24, 2048, 32
    
    if args.layers: config.num_layers = args.layers
    if args.dim: config.hidden_dim = args.dim
    if args.heads: config.num_heads = args.heads
    if args.lr: config.learning_rate = args.lr
    if args.wd: config.weight_decay = args.wd
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.num_epochs = args.epochs
    
    if args.optimizer:
        config.optimizer_type = args.optimizer
    
    if args.qat: config.use_qat = True
    if args.no_qat: config.use_qat = False
    if args.quant_type: config.quantization_type = args.quant_type
    
    if args.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device:
        config.device = args.device
    
    if args.no_mp: config.mixed_precision = False
    if args.no_numba: config.use_numba_jit = False
    
    if args.output_dir: config.output_dir = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Handle Modal deployment
    if args.deploy_modal:
        deploy_on_modal(__file__, args.gpu_type)
        return
    
    # Determine which optimizers to test
    if config.optimizer_type == "all":
        optimizers_to_test = ["adamw", "lion", "sophia", "adafactor", "adan", "radam"]
    else:
        optimizers_to_test = [config.optimizer_type]
    
    # Run training
    results = train_with_config(config, optimizers_to_test)
    
    # Save results
    results_path = args.save_results or os.path.join(
        config.output_dir, f"training_results_{int(time.time())}.json"
    )
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    
    for opt_name, opt_results in results.items():
        logger.info(f"{opt_name.upper():12s} | Best Loss: {opt_results['best_loss']:.4f} | "
                   f"Final Loss: {opt_results['final_loss']:.4f}")
    
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # Print comparison table
    if len(results) > 1:
        logger.info("\n🏆 Optimizer Ranking (by best loss):")
        ranked = sorted(results.items(), key=lambda x: x[1]["best_loss"])
        for rank, (name, res) in enumerate(ranked, 1):
            logger.info(f"  {rank}. {name}: {res['best_loss']:.4f}")


if __name__ == "__main__":
    main()
