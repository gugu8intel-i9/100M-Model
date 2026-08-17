#!/usr/bin/env python3
"""
Visualize Optimizer Benchmark Results from Modal Cloud GPU
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Add font support
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
    plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
except:
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
        plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'DejaVu Sans']
    except:
        pass  # Use default fonts
plt.rcParams['axes.unicode_minus'] = False

def visualize_results(results_path: str, output_path: str = None):
    """Create visualization of optimizer benchmark results."""
    
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    ranking = data.get('ranking', [])
    config = data.get('benchmark_config', {})
    
    # Extract data for plotting
    optimizers = []
    best_losses = []
    final_losses = []
    times = []
    converged = []
    
    for name in ranking:
        if name in results and 'error' not in results[name]:
            res = results[name]
            if res.get('best_loss') != float('inf'):
                optimizers.append(name.upper())
                best_losses.append(res.get('best_loss', 0))
                final_losses.append(res.get('final_loss', 0))
                times.append(res.get('time_seconds', 0))
                converged.append(res.get('converged', False))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # Color palette
    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Plot 1: Best Loss Comparison (Bar Chart)
    ax1 = axes[0]
    bars1 = ax1.bar(optimizers, best_losses, color=colors[:len(optimizers)], 
                    edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Loss (↓ better)', fontsize=12, fontweight='bold')
    ax1.set_title('🏆 Best Loss by Optimizer\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_ylim(min(best_losses) * 0.95, max(best_losses) * 1.05)
    
    # Add value labels on bars
    for bar, loss in zip(bars1, best_losses):
        height = bar.get_height()
        ax1.annotate(f'{loss:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight winner
    if len(bars1) > 0:
        bars1[0].set_edgecolor('#FF4500')
        bars1[0].set_linewidth(3)
    
    # Plot 2: Training Time Comparison
    ax2 = axes[1]
    bars2 = ax2.bar(optimizers, times, color='#45B7D1', 
                    edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('⏱️ Training Time by Optimizer\n(Faster is Better)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, t in zip(bars2, times):
        height = bar.get_height()
        ax2.annotate(f'{t:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Loss vs Speed Scatter (Efficiency)
    ax3 = axes[2]
    scatter_colors = colors[:len(optimizers)]
    for i, (opt, loss, t) in enumerate(zip(optimizers, best_losses, times)):
        size = 300 if i == 0 else 200  # Winner bigger
        ax3.scatter(t, loss, s=size, c=scatter_colors[i], 
                   edgecolors='black', linewidths=2, label=opt, alpha=0.8)
        
        # Add label
        offset_x, offset_y = 5, 5 if i % 2 == 0 else -15
        ax3.annotate(opt, (t, loss), xytext=(offset_x, offset_y),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Training Time (seconds) → Faster', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Best Loss → Lower', fontsize=12, fontweight='bold')
    ax3.set_title('📊 Efficiency: Speed vs Quality\n(Ideal: Bottom-Left)', fontsize=14, fontweight='bold')
    
    # Add "ideal zone" annotation
    ax3.axhline(y=min(best_losses) * 1.02, color='green', linestyle='--', alpha=0.5, label='Target Loss')
    ax3.axvline(x=min(times) * 1.1, color='blue', linestyle='--', alpha=0.5, label='Target Speed')
    ax3.legend(loc='upper right', fontsize=8)
    
    # Main title
    hardware_info = config.get('hardware', {})
    gpu_name = hardware_info.get('gpu', 'GPU')
    total_time = config.get('total_time_seconds', 0)
    
    fig.suptitle(
        f'🚀 LLM Optimizer Benchmark Results\n'
        f'Hardware: {gpu_name} | Total Benchmark: {total_time:.1f}s | '
        f'Model: ~{config.get("model_params", {}).get("hidden_dim", "?")}d × {config.get("model_params", {}).get("layers", "?")}L',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    # Save figure
    if output_path is None:
        output_path = results_path.replace('.json', '_visualization.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import glob
    
    # Find latest benchmark results
    results_files = sorted(glob.glob('/home/z/my-project/download/optimizer_benchmark_*.json'))
    
    if not results_files:
        print("❌ No benchmark results found!")
    else:
        latest = results_files[-1]
        print(f"📊 Visualizing: {latest}")
        visualize_results(latest)
