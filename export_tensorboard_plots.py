#!/usr/bin/env python3
"""
Script to export TensorBoard plots as images.
Reads TensorBoard event files and generates PNG plots.
"""

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    data = {}
    # Get all scalar tags
    tags = ea.Tags()['scalars']

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}

    return data

def plot_metric(data, metric_name, save_path, smooth=0):
    """Plot a single metric and save to file."""
    if metric_name not in data:
        print(f"Metric '{metric_name}' not found in data")
        return

    steps = data[metric_name]['steps']
    values = data[metric_name]['values']

    # Apply smoothing if requested
    if smooth > 0 and len(values) > smooth:
        kernel = np.ones(smooth) / smooth
        values_smooth = np.convolve(values, kernel, mode='valid')
        steps_smooth = steps[smooth-1:]
    else:
        values_smooth = values
        steps_smooth = steps

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, alpha=0.3, label='Raw')
    plt.plot(steps_smooth, values_smooth, linewidth=2, label='Smoothed')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_multiple_runs(log_dirs, metric_name, save_path, smooth=0):
    """Plot the same metric from multiple runs for comparison."""
    plt.figure(figsize=(12, 7))

    for log_dir in log_dirs:
        run_name = Path(log_dir).parent.name + "/" + Path(log_dir).name
        data = load_tensorboard_data(log_dir)

        if metric_name not in data:
            print(f"Metric '{metric_name}' not found in {run_name}")
            continue

        steps = data[metric_name]['steps']
        values = data[metric_name]['values']

        # Apply smoothing
        if smooth > 0 and len(values) > smooth:
            kernel = np.ones(smooth) / smooth
            values = np.convolve(values, kernel, mode='valid')
            steps = steps[smooth-1:]

        plt.plot(steps, values, linewidth=2, label=run_name, alpha=0.8)

    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f'{metric_name} - Comparison')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {save_path}")

def main():
    # Configuration
    logs_base_dir = "logs/rsl_rl"
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Find all event files
    event_files = glob.glob(f"{logs_base_dir}/**/events.out.tfevents.*", recursive=True)

    if not event_files:
        print(f"No TensorBoard event files found in {logs_base_dir}")
        return

    # Get unique run directories
    run_dirs = list(set([os.path.dirname(f) for f in event_files]))
    run_dirs.sort()

    print(f"Found {len(run_dirs)} training runs")

    # Common metrics to export
    common_metrics = [
        'Loss/value_function',
        'Loss/surrogate',
        'Loss/learning_rate',
        'Train/mean_reward',
        'Train/mean_episode_length',
        'Policy/mean_noise_std',
    ]

    # Export plots for each run
    for run_dir in run_dirs:
        run_name = Path(run_dir).parent.name + "_" + Path(run_dir).name
        print(f"\nProcessing run: {run_name}")

        try:
            data = load_tensorboard_data(run_dir)
            print(f"  Available metrics: {len(data)}")

            # Print all available metrics
            print(f"  Metrics: {', '.join(list(data.keys())[:10])}...")

            # Create subdirectory for this run
            run_output_dir = os.path.join(output_dir, run_name)
            os.makedirs(run_output_dir, exist_ok=True)

            # Export all available metrics
            for metric in data.keys():
                safe_metric_name = metric.replace('/', '_').replace(' ', '_')
                save_path = os.path.join(run_output_dir, f"{safe_metric_name}.png")
                plot_metric(data, metric, save_path, smooth=10)

        except Exception as e:
            print(f"  Error processing {run_name}: {e}")

    # Export comparison plots for velocity tracking runs
    print("\nCreating comparison plots...")
    velocity_runs = [d for d in run_dirs if 'evobot_v1_velocity' in d]

    if len(velocity_runs) > 1:
        for metric in common_metrics:
            safe_metric_name = metric.replace('/', '_').replace(' ', '_')
            save_path = os.path.join(output_dir, f"comparison_{safe_metric_name}.png")
            try:
                plot_multiple_runs(velocity_runs, metric, save_path, smooth=10)
            except Exception as e:
                print(f"  Error creating comparison for {metric}: {e}")

    print(f"\n✓ All plots exported to: {output_dir}/")
    print(f"  Total runs processed: {len(run_dirs)}")

if __name__ == "__main__":
    main()
