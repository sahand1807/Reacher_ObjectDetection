"""
Plot custom training metrics including success rate from callbacks.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not installed!")
    sys.exit(1)


def smooth_curve(values, weight=0.9):
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_logs(log_dir='logs/sac'):
    """Load all metrics from latest training run."""
    log_path = Path(log_dir)
    run_dirs = sorted([d for d in log_path.glob("SAC_*") if d.is_dir()])
    
    if not run_dirs:
        print(f"No training runs found in {log_dir}")
        return None, None
    
    latest_run = run_dirs[-1]
    print(f"Loading: {latest_run.name}")
    
    event_files = list(latest_run.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None
    
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    
    # Extract all scalar metrics
    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    return metrics, latest_run.name


def plot_key_metrics(metrics, run_name, output_dir='results/sac'):
    """Plot the most important metrics for this project."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create 1x3 grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Metric selections with fallbacks
    plot_configs = [
        {
            'keys': ['rollout/success_rate_100ep', 'eval/success_rate'],
            'title': 'Success Rate (100 episodes)',
            'ylabel': 'Success Rate',
            'baseline': 0.85,
            'baseline_label': 'V1 Baseline (85%)',
            'color': '#2ecc71'
        },
        {
            'keys': ['rollout/mean_reward_100ep', 'rollout/ep_rew_mean'],
            'title': 'Mean Episode Reward (100 episodes)',
            'ylabel': 'Reward',
            'color': '#3498db'
        },
        {
            'keys': ['rollout/mean_distance_100ep', 'rollout/ep_final_distance_cm'],
            'title': 'Mean Distance to Target (100 episodes)',
            'ylabel': 'Distance (cm)',
            'color': '#e74c3c'
        }
    ]

    for idx, config in enumerate(plot_configs):
        ax = axes[idx]
        
        # Find first available metric key
        data = None
        for key in config['keys']:
            if key in metrics:
                data = metrics[key]
                break
        
        if data is None:
            ax.text(0.5, 0.5, 'Metric not available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(config['title'], fontsize=12, fontweight='bold')
            continue
        
        steps = np.array(data['steps'])
        values = np.array(data['values'])
        
        # Plot raw data (light)
        ax.plot(steps, values, alpha=0.2, linewidth=1, 
               color=config['color'], label='Raw')
        
        # Plot smoothed
        if len(values) > 10:
            smoothed = smooth_curve(values, weight=0.9)
            ax.plot(steps, smoothed, linewidth=2.5, 
                   color=config['color'], label='Smoothed')
        
        # Add baseline if specified
        if 'baseline' in config:
            ax.axhline(y=config['baseline'], color='red', 
                      linestyle='--', linewidth=2,
                      label=config['baseline_label'], alpha=0.7)
            
            # Print final value vs baseline
            if len(values) > 0:
                final_val = values[-1]
                print(f"\n{config['title']}:")
                print(f"  Final: {final_val*100:.1f}%")
                print(f"  Baseline: {config['baseline']*100:.1f}%")
                if final_val >= config['baseline']:
                    print(f"  ✓ Exceeds baseline by {(final_val - config['baseline'])*100:.1f}%")
                else:
                    print(f"  ✗ Below baseline by {(config['baseline'] - final_val)*100:.1f}%")
        
        ax.set_xlabel('Timesteps', fontsize=11, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=11, fontweight='bold')
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        
        # Format x-axis
        ax.ticklabel_format(style='plain', axis='x')
    
    fig.suptitle(f'Training Metrics - {run_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = output_path / f'training_metrics_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_path}")
    
    return save_path


def plot_evaluation_progress(metrics, run_name, output_dir='results/sac'):
    """Plot evaluation metrics over time."""
    output_path = Path(output_dir)
    
    # Check if we have eval metrics
    eval_keys = ['eval/success_rate', 'eval/mean_reward', 'eval/mean_distance_cm']
    eval_data = {k: metrics[k] for k in eval_keys if k in metrics}
    
    if not eval_data:
        print("\nNo evaluation metrics found")
        return None
    
    fig, axes = plt.subplots(1, len(eval_data), figsize=(6*len(eval_data), 5))
    if len(eval_data) == 1:
        axes = [axes]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    titles = ['Evaluation Success Rate', 'Evaluation Mean Reward', 'Evaluation Mean Distance']
    
    for idx, (key, data) in enumerate(eval_data.items()):
        ax = axes[idx]
        steps = np.array(data['steps'])
        values = np.array(data['values'])
        
        ax.plot(steps, values, 'o-', linewidth=2, markersize=6,
               color=colors[idx], label='Evaluation')
        
        # Add baseline for success rate
        if 'success' in key:
            ax.axhline(y=0.85, color='red', linestyle='--', 
                      linewidth=2, label='V1 Baseline', alpha=0.7)
        
        ax.set_xlabel('Timesteps', fontsize=11, fontweight='bold')
        ax.set_ylabel(titles[idx].split()[-1], fontsize=11, fontweight='bold')
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = output_path / f'evaluation_progress_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")
    
    return save_path


def main():
    """Main entry point."""
    print("="*70)
    print("TRAINING METRICS PLOTTER")
    print("="*70)
    
    # Load metrics
    metrics, run_name = load_logs()
    
    if metrics is None:
        print("Failed to load logs!")
        return
    
    print(f"Loaded {len(metrics)} metrics")
    
    # Generate plots
    print("\n" + "-"*70)
    print("Generating Plots")
    print("-"*70)
    
    plot_key_metrics(metrics, run_name)
    plot_evaluation_progress(metrics, run_name)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
