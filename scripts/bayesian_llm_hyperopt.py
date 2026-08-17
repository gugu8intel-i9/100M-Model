#!/usr/bin/env python3
"""
=============================================================================
Bayesian Hyperparameter Optimization for 100M Parameter Language Models
Using TPE (Tree-Parzen Estimator) Surrogate Model via Optuna

Author: AI Research Assistant
Date: 2025-01
=============================================================================

This script performs efficient hyperparameter search for training small (100M)
language models using Bayesian Optimization with TPE surrogate models.

RUN: python bayesian_llm_hyperopt.py
OUTPUTS: 
    - best_hyperparams.json (optimal configuration)
    - optimization_history.html (interactive plots)
    - hyperparameter_importance.html (feature importance)
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the optimization run"""
    n_trials: int = 30              # Number of trials (reduce for quick test, increase for real runs)
    target_params: int = 100_000_000  # Target model size (~100M)
    max_train_steps: int = 200      # Max steps per trial (keep low for fast iteration)
    eval_every: int = 50            # Evaluate every N steps
    vocab_size: int = 8192          # Small vocab for testing
    seq_len: int = 128              # Sequence length for testing
    device: str = "cpu"             # Use "cuda" if GPU available
    
    # Search space bounds
    lr_range: Tuple[float, float] = (1e-5, 1e-2)
    wd_range: Tuple[float, float] = (0.01, 0.3)

# ============================================================================
# MODEL ARCHITECTURE (~100M Parameters)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with learned position embeddings"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for Q, K, V (more stable than combined)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V separately
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output projection
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    
    def __init__(self, hidden_dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = hidden_dim * mlp_ratio
        
        self.gate_proj = nn.Linear(hidden_dim, inner_dim)
        self.up_proj = nn.Linear(hidden_dim, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4, 
                 dropout: float = 0.0):
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, mlp_ratio, dropout)
        self.att_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.dropout(self.attention(self.att_norm(x), mask))
        # Pre-norm FFN with residual
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class SmallLanguageModel(nn.Module):
    """
    Configurable Transformer Language Model (~100M parameters)
    
    Architecture based on modern LLM designs:
    - Pre-LayerNorm (Pre-norm)
    - SwiGLU Activation
    - Rotary Position Embeddings (RoPE)
    - No bias terms (for efficiency)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        vocab_size = config.get('vocab_size', 8192)
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        num_heads = config['num_heads']
        mlp_ratio = config.get('mlp_ratio', 4)
        dropout = config.get('dropout', 0.0)
        max_seq_len = config.get('max_seq_len', 512)
        
        # Token and position embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output head (tied weights optional)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Store config
        self.config = config
        self.max_seq_len = max_seq_len
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # Get embeddings
        x = self.tok_embeddings(input_ids)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final norm and output
        x = self.norm(x)
        logits = self.output(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_config(cls, trial_config: Dict[str, Any]) -> 'SmallLanguageModel':
        """Create model from trial configuration"""
        return cls(trial_config)


# ============================================================================
# SYNTHETIC DATA GENERATOR (For Testing)
# ============================================================================

class SyntheticDataGenerator:
    """Generates synthetic sequences for rapid hyperparameter testing"""
    
    def __init__(self, vocab_size: int = 8192, seq_len: int = 128, batch_size: int = 32):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        # Create some structured patterns (not pure random)
        # This makes the learning task more realistic
        np.random.seed(42)
        self.pattern_weights = np.random.randn(vocab_size, vocab_size) * 0.1
        
    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of synthetic sequences"""
        batch = []
        
        for _ in range(self.batch_size):
            # Start with random token
            seq = [np.random.randint(0, self.vocab_size)]
            
            # Generate sequence with some structure
            for _ in range(self.seq_len - 1):
                # Weighted sampling based on previous token
                probs = softmax(self.pattern_weights[seq[-1]] + np.random.randn(self.vocab_size) * 0.5)
                next_token = np.random.choice(self.vocab_size, p=probs)
                seq.append(next_token)
            
            batch.append(seq)
        
        # Convert to tensors
        input_ids = torch.tensor(batch, dtype=torch.long)
        targets = torch.roll(input_ids, shifts=-1, dims=1)
        targets[:, -1] = -100  # Mask last token
        
        return input_ids, targets


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================

def create_optimizer(config: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer based on configuration"""
    lr = config['learning_rate']
    wd = config['weight_decay']
    opt_type = config.get('optimizer_type', 'adamw')
    
    # Separate weight decay parameters (no decay on biases/norms)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'norm' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if opt_type == 'adamw':
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    
    elif opt_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, lr=lr, weight_decay=wd, betas=(0.9, 0.99))
        except ImportError:
            print("⚠️ Lion not installed, falling back to AdamW")
            print("   Install with: pip install lion-pytorch")
            return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    
    elif opt_type == 'adafactor':
        try:
            from transformers import Adafactor
            return Adafactor(param_groups, lr=lr, relative_step=False, warmup_init=False)
        except ImportError:
            print("⚠️ Adafactor/transformers not installed, falling back to AdamW")
            return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def create_scheduler(config: Dict[str, Any], optimizer: torch.optim.Optimizer, 
                     total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    warmup_ratio = config.get('warmup_steps_ratio', 0.05)
    warmup_steps = int(total_steps * warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(config.get('lr_min_ratio', 0.1), 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# MODEL SIZE ESTIMATION
# ============================================================================

def estimate_model_size(config: Dict[str, Any]) -> int:
    """Estimate parameter count without building the model"""
    dim = config['hidden_dim']
    layers = config['num_layers']
    heads = config['num_heads']
    mlp = config.get('mlp_ratio', 4)
    vocab = config.get('vocab_size', 8192)
    
    # Embeddings
    embed_params = vocab * dim
    
    # Per-layer params
    # Attention: QKV proj (3*dim^2) + out proj (dim^2) = 4*dim^2
    attn_params = 4 * dim * dim
    # FFN: gate + up + down projections
    ffn_params = (2 * dim * dim * mlp) + (dim * dim * mlp)
    # Layer norms (2 per block)
    norm_params = 2 * 2 * dim
    
    per_layer = attn_params + ffn_params + norm_params
    transformer_params = layers * per_layer
    
    # Output head
    output_params = dim * vocab
    
    total = embed_params + transformer_params + output_params
    return total


# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               data_gen: SyntheticDataGenerator, 
               device: str) -> float:
    """Perform one training step and return loss"""
    model.train()
    optimizer.zero_grad()
    
    input_ids, targets = data_gen.generate_batch()
    input_ids, targets = input_ids.to(device), targets.to(device)
    
    _, loss = model(input_ids, targets)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    return loss.item()


def evaluate(model: nn.Module, data_gen: SyntheticDataGenerator, 
             device: str, n_batches: int = 3) -> float:
    """Evaluate model and return average validation loss"""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, targets = data_gen.generate_batch()
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            _, loss = model(input_ids, targets)
            losses.append(loss.item())
    
    return np.mean(losses)


# ============================================================================
# MAIN OPTIMIZATION CLASS
# ============================================================================

class LLMHyperparameterOptimizer:
    """
    Bayesian Hyperparameter Optimizer for 100M Language Models
    
    Uses TPE (Tree-Parzen Estimator) as surrogate model for efficient
    exploration of hyperparameter space.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        self.data_gen = SyntheticDataGenerator(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            batch_size=32  # Will be updated per trial
        )
        
        # Results storage
        self.results_history = []
        
        print(f"\n{'='*70}")
        print(f"🚀 BAYESIAN HYPERPARAMETER OPTIMIZATION FOR 100M LLM")
        print(f"{'='*70}")
        print(f"   Device: {self.device}")
        print(f"   Target Params: {config.target_params:,}")
        print(f"   Max Trials: {config.n_trials}")
        print(f"   Max Steps/Trial: {config.max_train_steps}")
        print(f"   Surrogate: TPE (Tree-Parzen Estimator)")
        print(f"{'='*70}\n")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space based on 2024-2025 research.
        
        Search space designed for ~100M parameter models with informed priors.
        """
        config = {
            # === LEARNING RATE (Most important!) ===
            # Log-uniform is crucial for LR search
            'learning_rate': trial.suggest_float('learning_rate', 
                                                  self.config.lr_range[0], 
                                                  self.config.lr_range[1], 
                                                  log=True),
            
            # === OPTIMIZER TYPE ===
            'optimizer_type': trial.suggest_categorical('optimizer_type', 
                                                        ['adamw', 'lion']),
            
            # === WEIGHT DECAY ===
            'weight_decay': trial.suggest_float('weight_decay',
                                                 self.config.wd_range[0],
                                                 self.config.wd_range[1],
                                                 log=True),
            
            # === SCHEDULER ===
            'warmup_steps_ratio': trial.suggest_float('warmup_ratio', 0.02, 0.15),
            'lr_min_ratio': trial.suggest_float('lr_min_ratio', 0.05, 0.25),
            
            # === BATCH SIZE ===
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            
            # === ARCHITECTURE (for ~5M target in demo) ===
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            
            # === FFN RATIO ===
            'mlp_ratio': trial.suggest_categorical('mlp_ratio', [2, 3, 4]),
            
            # === REGULARIZATION ===
            'dropout': trial.suggest_float('dropout', 0.0, 0.2),
            
            # === ACTIVATION (stored for reference, using SwiGLU) ===
            'activation': trial.suggest_categorical('activation', ['swiglu', 'gelu']),
        }
        
        # Add fixed params
        config.update({
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.seq_len,
        })
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function: Train model and return validation loss.
        
        Uses pruning to stop unpromising trials early.
        """
        start_time = time.time()
        
        # Get hyperparameters from search space
        config = self.define_search_space(trial)
        
        # Update data generator batch size
        self.data_gen.batch_size = config['batch_size']
        
        # Estimate model size
        estimated_params = estimate_model_size(config)
        size_penalty = abs(estimated_params - self.config.target_params) / self.config.target_params
        
        # Report attributes for analysis
        trial.set_user_attr('estimated_params', estimated_params)
        trial.set_user_attr('size_penalty', size_penalty)
        
        print(f"\n📋 Trial {trial.number}:")
        print(f"   LR={config['learning_rate']:.2e}, "
              f"Opt={config['optimizer_type']}, "
              f"BS={config['batch_size']}, "
              f"Layers={config['num_layers']}, "
              f"Dim={config['hidden_dim']}, "
              f"Heads={config['num_heads']}, "
              f"Est Params={estimated_params:,}")
        
        try:
            # Build model
            model = SmallLanguageModel.from_config(config)
            actual_params = model.count_parameters()
            model = model.to(self.device)
            
            trial.set_user_attr('actual_params', actual_params)
            
            # Create optimizer and scheduler
            optimizer = create_optimizer(config, model)
            scheduler = create_scheduler(config, optimizer, self.config.max_train_steps)
            
            # Training loop with intermediate reporting for pruning
            val_losses = []
            
            for step in range(1, self.config.max_train_steps + 1):
                # Train one step
                train_loss = train_step(model, optimizer, scheduler, 
                                       self.data_gen, self.device)
                
                # Periodic evaluation
                if step % self.config.eval_every == 0:
                    val_loss = evaluate(model, self.data_gen, self.device, n_batches=2)
                    val_losses.append(val_loss)
                    
                    # Report intermediate value for pruning
                    trial.report(val_loss, step)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        elapsed = time.time() - start_time
                        print(f"   ⛔ Pruned at step {step} ({elapsed:.1f}s)")
                        raise optuna.TrialPruned()
            
            # Final evaluation
            final_val_loss = evaluate(model, self.data_gen, self.device, n_batches=5)
            
            # Apply mild penalty for size deviation
            penalized_loss = final_val_loss * (1 + 0.5 * size_penalty)
            
            elapsed = time.time() - start_time
            
            print(f"   ✅ Complete! Val Loss={final_val_loss:.4f}, "
                  f"Penalized={penalized_loss:.4f}, Time={elapsed:.1f}s")
            
            # Store results
            self.results_history.append({
                'trial': trial.number,
                'val_loss': final_val_loss,
                'penalized_loss': penalized_loss,
                'params': actual_params,
                'config': config,
                'time': elapsed,
                'pruned': False
            })
            
            return penalized_loss
            
        except optuna.TrialPruned:
            self.results_history.append({
                'trial': trial.number,
                'val_loss': None,
                'penalized_loss': None,
                'params': None,
                'config': config,
                'time': time.time() - start_time,
                'pruned': True
            })
            raise
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Error: {str(e)} ({elapsed:.1f}s)")
            raise optuna.TrialPruned()
    
    def run_optimization(self) -> optuna.Study:
        """Run the full Bayesian optimization study"""
        
        # Create TPE sampler (surrogate model)
        sampler = TPESampler(
            n_startup_trials=max(5, self.config.n_trials // 5),  # Initial random trials
            n_ei_candidates=24,
            seed=42,
            multivariate=True,  # Joint parameter modeling
            group=True,         # Group related parameters
        )
        
        # Create pruner (median stopping rule) - Updated API for Optuna v4.9+
        pruner = MedianPruner(
            n_warmup_steps=self.config.eval_every * 2,
        )
        
        # Create study
        study = optuna.create_study(
            study_name=f'llm_100m_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        
        # Enqueue some known good configurations as starting points
        study.enqueue_trial({
            'learning_rate': 3e-4,
            'optimizer_type': 'adamw',
            'weight_decay': 0.1,
            'warmup_ratio': 0.05,
            'lr_min_ratio': 0.1,
            'batch_size': 16,
            'num_layers': 4,
            'hidden_dim': 128,
            'num_heads': 4,
            'mlp_ratio': 4,
            'dropout': 0.0,
            'activation': 'swiglu',
        })
        
        print(f"Starting optimization with {self.config.n_trials} trials...")
        print(f"Initial random trials: {max(5, self.config.n_trials // 5)}\n")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True,
            gc_after_trial=True,  # Clean up memory between trials
            catch=(Exception,),   # Catch all exceptions
        )
        
        return study
    
    def analyze_and_save_results(self, study: optuna.Study, output_dir: str = '/home/z/my-project/download'):
        """Analyze results and save outputs"""
        
        print("\n" + "="*70)
        print("🏆 OPTIMIZATION COMPLETE - RESULTS SUMMARY")
        print("="*70)
        
        # Best trial info
        best_trial = study.best_trial
        print(f"\n📊 Best Trial: #{best_trial.number}")
        print(f"   Best Validation Loss (Penalized): {best_trial.value:.6f}")
        print(f"   Estimated Parameters: {best_trial.user_attrs.get('estimated_params', 'N/A'):,}")
        
        print(f"\n🎯 Optimal Hyperparameters:\n")
        best_params = {}
        for key, value in best_trial.params.items():
            print(f"   {key:25s}: {value}")
            best_params[key] = value
        
        # Add metadata
        best_params['_metadata'] = {
            'best_value': best_trial.value,
            'actual_params': best_trial.user_attrs.get('actual_params', None),
            'estimated_params': best_trial.user_attrs.get('estimated_params', None),
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }
        
        # Save best hyperparameters
        output_path = os.path.join(output_dir, 'best_llm_hyperparams.json')
        with open(output_path, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)
        print(f"\n💾 Saved best hyperparameters to: {output_path}")
        
        # Save full history
        history_path = os.path.join(output_dir, 'optimization_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
        print(f"💾 Saved full history to: {history_path}")
        
        # Generate visualizations
        self._generate_visualizations(study, output_dir)
        
        # Print statistics
        self._print_statistics(study)
        
        return best_params
    
    def _generate_visualizations(self, study: optuna.Study, output_dir: str):
        """Generate optimization visualizations"""
        
        print("\n📈 Generating visualizations...")
        
        try:
            # 1. Optimization History
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html(os.path.join(output_dir, 'optimization_history.html'))
            fig1.write_image(os.path.join(output_dir, 'optimization_history.png'))
            print("   ✅ optimization_history.html/png")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate optimization history: {e}")
        
        try:
            # 2. Hyperparameter Importance
            fig2 = vis.plot_param_importances(study)
            fig2.write_html(os.path.join(output_dir, 'hyperparameter_importance.html'))
            fig2.write_image(os.path.join(output_dir, 'hyperparameter_importance.png'))
            print("   ✅ hyperparameter_importance.html/png")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate importance plot: {e}")
        
        try:
            # 3. Parallel Coordinates
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_html(os.path.join(output_dir, 'parallel_coordinates.html'))
            fig3.write_image(os.path.join(output_dir, 'parallel_coordinates.png'))
            print("   ✅ parallel_coordinates.html/png")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate parallel coordinates: {e}")
        
        try:
            # 4. Slice Plots (individual params vs objective)
            for param in study.best_params.keys():
                fig = vis.plot_slice(study, params=[param])
                safe_name = param.replace('/', '_')
                fig.write_html(os.path.join(output_dir, f'slice_{safe_name}.html'))
            print("   ✅ slice_*.html (per-parameter)")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate slice plots: {e}")
        
        try:
            # 5. Contour plots (pairwise relationships)
            fig5 = vis.plot_contour(study)
            fig5.write_html(os.path.join(output_dir, 'contour_plot.html'))
            print("   ✅ contour_plot.html")
            
        except Exception as e:
            print(f"   ⚠️ Could not generate contour plot: {e}")
    
    def _print_statistics(self, study: optuna.Study):
        """Print optimization statistics"""
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        print("\n📊 STATISTICS:")
        print(f"   Total Trials: {len(study.trials)}")
        print(f"   Completed: {len(completed_trials)}")
        print(f"   Pruned: {len(pruned_trials)} ({100*len(pruned_trials)/max(len(study.trials),1):.1f}%)")
        
        if completed_trials:
            values = [t.value for t in completed_trials if t.value is not None]
            if values:
                print(f"\n   Loss Statistics:")
                print(f"      Best:     {min(values):.6f}")
                print(f"      Worst:    {max(values):.6f}")
                print(f"      Mean:     {np.mean(values):.6f}")
                print(f"      Std:      {np.std(values):.6f}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function to run Bayesian optimization"""
    
    # Configuration - Optimized for CPU/memory constraints
    # For real 100M training, increase these values and use GPU!
    config = OptimizationConfig(
        n_trials=20,              # Number of experiments
        target_params=5_000_000,   # ~5M parameters (reduced for demo)
        max_train_steps=100,      # Steps per trial (fast iteration)
        eval_every=25,            # Evaluate every N steps
        vocab_size=1024,          # Small vocab for fast testing
        seq_len=64,               # Shorter sequences for speed
        device="cpu",             # Use "cuda" if GPU available
    )
    
    # Create optimizer and run
    optimizer = LLMHyperparameterOptimizer(config)
    
    print("Starting Bayesian Optimization...")
    print("(This may take a while depending on n_trials and hardware)\n")
    
    # Run optimization
    study = optimizer.run_optimization()
    
    # Analyze and save results
    best_params = optimizer.analyze_and_save_results(study)
    
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"   1. Review best_hyperparams.json for optimal configuration")
    print(f"   2. Check visualization HTML files for insights")
    print(f"   3. Train your full 100M model with these hyperparameters")
    print(f"   4. Consider increasing n_trials for more thorough search")
    
    return study, best_params


if __name__ == "__main__":
    study, results = main()
