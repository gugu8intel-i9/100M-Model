#!/usr/bin/env python3
"""
=============================================================================
FAST Bayesian Hyperparameter Optimization Demo for 100M LLMs
Using TPE (Tree-Parzen Estimator) Surrogate Model via Optuna

This is a FAST DEMO version that simulates training to demonstrate
the Bayesian optimization workflow. For actual training, use the
full version with GPU access.

RUN: python bayesian_llm_hyperopt_fast.py
OUTPUTS: 
    - best_hyperparams.json (optimal configuration)
    - optimization_history.html (interactive plots)
    - hyperparameter_importance.html (feature importance)
=============================================================================
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import numpy as np
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass 
class FastOptimizationConfig:
    """Configuration for fast demo optimization"""
    n_trials: int = 25              # Number of trials
    target_params: int = 100_000_000  # Target ~100M parameters
    
    # Search space bounds (based on 2024-2025 research)
    lr_range: Tuple[float, float] = (1e-5, 1e-2)
    wd_range: Tuple[float, float] = (0.01, 0.3)


# ============================================================================
# SIMULATED OBJECTIVE FUNCTION (Replaces Actual Training)
# ============================================================================

def estimate_model_size(config: Dict[str, Any]) -> int:
    """Estimate parameter count from config"""
    dim = config['hidden_dim']
    layers = config['num_layers']
    heads = config['num_heads']
    mlp = config.get('mlp_ratio', 4)
    vocab = config.get('vocab_size', 32000)
    
    embed_params = vocab * dim
    attn_params = layers * 4 * dim * dim  # QKV + Out per layer
    ffn_params = layers * 2 * dim * dim * mlp  # FFN per layer
    output_params = dim * vocab
    
    return embed_params + attn_params + ffn_params + output_params


def simulated_training_loss(config: Dict[str, Any], noise: float = 0.05) -> float:
    """
    Simulate validation loss based on hyperparameters.
    
    This uses a known "good" region in hyperparameter space plus noise,
    mimicking what you'd see in real training.
    
    Based on research findings for LLM training:
    - Optimal LR typically around 1e-4 to 6e-4
    - AdamW generally most stable
    - Moderate weight decay (0.01-0.1) works best
    """
    
    lr = config['learning_rate']
    wd = config['weight_decay']
    batch_size = config['batch_size']
    num_layers = config['num_layers']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    optimizer_type = config['optimizer_type']
    dropout = config['dropout']
    mlp_ratio = config['mlp_ratio']
    
    # Base loss (simulates convergence)
    base_loss = 3.0  # Starting perplexity-like loss
    
    # Learning rate effect (optimal around 3e-4)
    optimal_lr = 3e-4
    lr_penalty = 0.5 * ((np.log10(lr) - np.log10(optimal_lr)) ** 2)
    
    # Optimizer preference (AdamW best, Lion good, others ok)
    optimizer_bonus = {'adamw': 0.0, 'lion': 0.05, 'adafactor': 0.15}.get(optimizer_type, 0.1)
    
    # Weight decay effect (moderate is good)
    optimal_wd = 0.1
    wd_penalty = 0.3 * ((np.log10(wd) - np.log10(optimal_wd)) ** 2)
    
    # Batch size effect (larger usually better up to a point)
    bs_penalty = 0.02 * (64 / batch_size) if batch_size < 64 else 0.01 * (batch_size / 64)
    
    # Architecture effects
    # Hidden dim should be divisible by heads properly
    head_dim = hidden_dim // num_heads
    arch_penalty = 0.1 if head_dim < 32 else 0.0  # Penalize small head dims
    
    # Layer/depth effect (deeper can be better but harder to train)
    depth_penalty = 0.02 * abs(num_layers - 12)  # Optimal around 12 layers for 100M
    
    # Dropout regularization effect
    drop_penalty = 0.1 * dropout if dropout > 0.15 else 0.0  # Too much dropout bad
    
    # MLP ratio effect (4x is standard and works well)
    mlp_penalty = 0.05 * abs(mlp_ratio - 4)
    
    # Size penalty (prefer closer to target)
    estimated_params = estimate_model_size(config)
    size_penalty = 0.001 * abs(estimated_params - 100_000_000) / 100_000_000
    
    # Combine all effects
    total_loss = (base_loss 
                  + lr_penalty 
                  + optimizer_bonus 
                  + wd_penalty 
                  + bs_penalty 
                  + arch_penalty 
                  + depth_penalty 
                  + drop_penalty 
                  + mlp_penalty 
                  + size_penalty)
    
    # Add realistic noise (training stochasticity)
    total_loss += np.random.normal(0, noise)
    
    return max(1.5, min(6.0, total_loss))  # Clamp to reasonable range


# ============================================================================
# MAIN OPTIMIZATION CLASS
# ============================================================================

class FastLLMOptimizer:
    """
    Fast Bayesian Hyperparameter Optimizer for 100M Language Models
    
    Uses TPE surrogate model with simulated training for quick demonstration.
    """
    
    def __init__(self, config: FastOptimizationConfig):
        self.config = config
        
        print(f"\n{'='*70}")
        print(f"🚀 FAST BAYESIAN HYPERPARAMETER OPTIMIZATION FOR 100M LLM")
        print(f"(Demo Mode - Simulated Training)")
        print(f"{'='*70}")
        print(f"   Target Params: {config.target_params:,}")
        print(f"   Max Trials: {config.n_trials}")
        print(f"   Surrogate: TPE (Tree-Parzen Estimator)")
        print(f"{'='*70}\n")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space based on 2024-2025 research.
        
        These ranges are designed for ~100M parameter models.
        """
        config = {
            # === LEARNING RATE (Most important hyperparameter!) ===
            'learning_rate': trial.suggest_float('learning_rate', 
                                                  self.config.lr_range[0], 
                                                  self.config.lr_range[1], 
                                                  log=True),
            
            # === OPTIMIZER TYPE ===
            'optimizer_type': trial.suggest_categorical('optimizer_type', 
                                                        ['adamw', 'lion', 'adafactor']),
            
            # === WEIGHT DECAY ===
            'weight_decay': trial.suggest_float('weight_decay',
                                                 self.config.wd_range[0],
                                                 self.config.wd_range[1],
                                                 log=True),
            
            # === SCHEDULER ===
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.02, 0.15),
            'lr_min_ratio': trial.suggest_float('lr_min_ratio', 0.05, 0.25),
            
            # === BATCH SIZE ===
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            
            # === ARCHITECTURE (for ~100M target) ===
            'num_layers': trial.suggest_int('num_layers', 8, 16),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [512, 768, 1024]),
            'num_heads': trial.suggest_categorical('num_heads', [8, 12, 16]),
            
            # === FFN RATIO ===
            'mlp_ratio': trial.suggest_categorical('mlp_ratio', [2, 3, 4]),
            
            # === REGULARIZATION ===
            'dropout': trial.suggest_float('dropout', 0.0, 0.25),
            
            # === ACTIVATION ===
            'activation': trial.suggest_categorical('activation', ['swiglu', 'gelu']),
            
            # Fixed params
            'vocab_size': 32000,
            'max_seq_len': 2048,
        }
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function using simulated training loss.
        
        In production, this would train an actual model.
        """
        start_time = time.time()
        
        # Get hyperparameters
        config = self.define_search_space(trial)
        
        # Estimate model size
        estimated_params = estimate_model_size(config)
        size_penalty = abs(estimated_params - self.config.target_params) / self.config.target_params
        
        # Store attributes
        trial.set_user_attr('estimated_params', estimated_params)
        trial.set_user_attr('size_ratio', estimated_params / self.config.target_params)
        
        print(f"\n📋 Trial {trial.number}:")
        print(f"   LR={config['learning_rate']:.2e}, "
              f"Opt={config['optimizer_type']}, "
              f"BS={config['batch_size']}, "
              f"WD={config['weight_decay']:.3f}")
        print(f"   Layers={config['num_layers']}, "
              f"Dim={config['hidden_dim']}, "
              f"Heads={config['num_heads']}, "
              f"MLP={config['mlp_ratio']}x")
        print(f"   Est Params={estimated_params:,} "
              f"({estimated_params/self.config.target_params*100:.1f}% of target)")
        
        # Simulate training iterations with pruning checkpoints
        n_steps = 20
        losses = []
        
        for step in range(1, n_steps + 1):
            # Simulate training progress (loss decreases over time)
            step_noise = 0.3 * (1 - step/n_steps)  # More noise early on
            val_loss = simulated_training_loss(config, noise=step_noise)
            
            # Apply mild size penalty
            penalized_loss = val_loss * (1 + 0.3 * size_penalty)
            
            losses.append(penalized_loss)
            
            # Report intermediate value for pruning
            trial.report(penalized_loss, step)
            
            # Check pruning
            if trial.should_prune():
                elapsed = time.time() - start_time
                print(f"   ⛔ Pruned at step {step}/{n_steps} ({elapsed:.2f}s)")
                raise optuna.TrialPruned()
            
            # Small delay to simulate computation
            time.sleep(0.05)
        
        final_loss = np.mean(losses[-5:])  # Use last 5 steps
        
        elapsed = time.time() - start_time
        print(f"   ✅ Complete! Loss={final_loss:.4f}, Time={elapsed:.2f}s")
        
        return final_loss
    
    def run_optimization(self) -> optuna.Study:
        """Run the full Bayesian optimization study"""
        
        # TPE Sampler (surrogate model)
        sampler = TPESampler(
            n_startup_trials=max(5, self.config.n_trials // 5),
            n_ei_candidates=24,
            seed=42,
            multivariate=True,
            group=True,
        )
        
        # Median pruner
        pruner = MedianPruner(
            n_warmup_steps=5,
        )
        
        # Create study
        study = optuna.create_study(
            study_name=f'llm_100m_fast_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        
        # Enqueue known good starting point (based on research)
        study.enqueue_trial({
            'learning_rate': 3e-4,
            'optimizer_type': 'adamw',
            'weight_decay': 0.1,
            'warmup_ratio': 0.05,
            'lr_min_ratio': 0.1,
            'batch_size': 128,
            'num_layers': 12,
            'hidden_dim': 768,
            'num_heads': 12,
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
            gc_after_trial=True,
            catch=(Exception,),
        )
        
        return study
    
    def analyze_and_save_results(self, study: optuna.Study, 
                                  output_dir: str = '/home/z/my-project/download'):
        """Analyze results and save outputs"""
        
        print("\n" + "="*70)
        print("🏆 OPTIMIZATION COMPLETE - RESULTS SUMMARY")
        print("="*70)
        
        # Best trial info
        best_trial = study.best_trial
        print(f"\n📊 Best Trial: #{best_trial.number}")
        print(f"   Best Loss: {best_trial.value:.6f}")
        print(f"   Estimated Parameters: {best_trial.user_attrs.get('estimated_params', 'N/A'):,}")
        
        print(f"\n🎯 Optimal Hyperparameters:\n")
        best_params = {}
        for key, value in best_trial.params.items():
            print(f"   {key:25s}: {value}")
            best_params[key] = value
        
        # Add metadata
        best_params['_metadata'] = {
            'best_value': best_trial.value,
            'estimated_params': int(best_trial.user_attrs.get('estimated_params', 0)),
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'note': 'Fast demo mode with simulated training. Use full version for real training.',
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'best_llm_hyperparams.json')
        with open(output_path, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)
        print(f"\n💾 Saved to: {output_path}")
        
        # Generate visualizations
        self._generate_visualizations(study, output_dir)
        
        # Statistics
        self._print_statistics(study)
        
        return best_params
    
    def _generate_visualizations(self, study: optuna.Study, output_dir: str):
        """Generate optimization visualizations"""
        
        print("\n📈 Generating visualizations...")
        
        try:
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html(os.path.join(output_dir, 'optimization_history.html'))
            fig1.write_image(os.path.join(output_dir, 'optimization_history.png'))
            print("   ✅ optimization_history.html/png")
        except Exception as e:
            print(f"   ⚠️ History plot error: {e}")
        
        try:
            fig2 = vis.plot_param_importances(study)
            fig2.write_html(os.path.join(output_dir, 'hyperparameter_importance.html'))
            fig2.write_image(os.path.join(output_dir, 'hyperparameter_importance.png'))
            print("   ✅ hyperparameter_importance.html/png")
        except Exception as e:
            print(f"   ⚠️ Importance plot error: {e}")
        
        try:
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_html(os.path.join(output_dir, 'parallel_coordinates.html'))
            fig3.write_image(os.path.join(output_dir, 'parallel_coordinates.png'))
            print("   ✅ parallel_coordinates.html/png")
        except Exception as e:
            print(f"   ⚠️ Parallel coords error: {e}")
        
        try:
            for param in list(study.best_params.keys())[:6]:  # Top 6 params
                fig = vis.plot_slice(study, params=[param])
                safe_name = param.replace('/', '_')
                fig.write_html(os.path.join(output_dir, f'slice_{safe_name}.html'))
            print("   ✅ slice_*.html (per-parameter)")
        except Exception as e:
            print(f"   ⚠️ Slice plots error: {e}")
        
        try:
            fig5 = vis.plot_contour(study, params=['learning_rate', 'weight_decay'])
            fig5.write_html(os.path.join(output_dir, 'contour_lr_wd.html'))
            print("   ✅ contour_lr_wd.html")
        except Exception as e:
            print(f"   ⚠️ Contour plot error: {e}")
    
    def _print_statistics(self, study: optuna.Study):
        """Print optimization statistics"""
        
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        print(f"\n📊 STATISTICS:")
        print(f"   Total Trials: {len(study.trials)}")
        print(f"   Completed: {len(completed)}")
        print(f"   Pruned: {len(pruned)} ({100*len(pruned)/max(len(study.trials),1):.1f}%)")
        
        if completed:
            values = [t.value for t in completed if t.value is not None]
            if values:
                print(f"\n   Loss Distribution:")
                print(f"      Best:     {min(values):.4f}")
                print(f"      Worst:    {max(values):.4f}")
                print(f"      Mean:     {np.mean(values):.4f}")
                print(f"      Std:      {np.std(values):.4f}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function to run fast Bayesian optimization demo"""
    
    config = FastOptimizationConfig(
        n_trials=25,
        target_params=100_000_000,
    )
    
    optimizer = FastLLMOptimizer(config)
    
    print("Starting FAST Bayesian Optimization Demo...")
    print("(Simulated training - no GPU required)\n")
    
    # Run optimization
    study = optimizer.run_optimization()
    
    # Analyze results
    best_params = optimizer.analyze_and_save_results(study)
    
    print("\n" + "="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files saved to: /home/z/my-project/download/")
    print(f"\n🔬 Next Steps:")
    print(f"   1. Open optimization_history.html to see convergence")
    print(f"   2. Check hyperparameter_importance.html for key factors")
    print(f"   3. Review best_hyperparams.json for your optimal config")
    print(f"   4. Use these hyperparameters with your real 100M model training")
    print(f"\n💡 For real training, use the full script with GPU access!")
    
    return study, best_params


if __name__ == "__main__":
    study, results = main()
