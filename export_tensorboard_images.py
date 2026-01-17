#!/usr/bin/env python3
"""
Script to export TensorBoard data as images and plots.
Usage: python export_tensorboard_images.py [--logdir PATH] [--output PATH]
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(logdir):
    """Load all scalar data from TensorBoard event files."""
    print(f"Loading TensorBoard data from: {logdir}")

    # Find all event files recursively
    event_files = list(Path(logdir).rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"No TensorBoard event files found in {logdir}")
        return {}

    print(f"Found {len(event_files)} event file(s)")

    all_data = defaultdict(lambda: defaultdict(list))

    for event_file in event_files:
        run_name = event_file.parent.name
        task_name = event_file.parent.parent.name
        full_run_name = f"{task_name}/{run_name}"

        print(f"  Loading: {full_run_name}")

        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()

            # Get all scalar tags
            scalar_tags = ea.Tags()['scalars']

            for tag in scalar_tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                all_data[tag][full_run_name] = {
                    'steps': steps,
                    'values': values
                }

        except Exception as e:
            print(f"    Error loading {event_file}: {e}")
            continue

    return all_data


def plot_scalar(tag, runs_data, output_dir, smooth_factor=0.6):
    """Plot a single scalar metric across all runs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for run_name, data in runs_data.items():
        steps = np.array(data['steps'])
        values = np.array(data['values'])

        # Plot raw data with transparency
        ax.plot(steps, values, alpha=0.3, linewidth=0.5)

        # Plot smoothed data
        if len(values) > 1:
            smoothed = exponential_moving_average(values, smooth_factor)
            ax.plot(steps, smoothed, label=run_name, linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title(tag)
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, alpha=0.3)

    # Save figure
    safe_tag = tag.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f"{safe_tag}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_path}")


def exponential_moving_average(values, smooth_factor=0.6):
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0]

    for value in values:
        smoothed_val = last * smooth_factor + (1 - smooth_factor) * value
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def create_comparison_plot(all_data, output_dir, metrics_to_compare=None):
    """Create comparison plots for key metrics across all runs."""

    if metrics_to_compare is None:
        # Default key metrics to compare
        metrics_to_compare = [
            'Loss/policy',
            'Loss/value',
            'Rewards/episode_reward',
            'Train/mean_reward',
            'Train/mean_episode_length',
        ]

    # Filter to metrics that exist in the data
    available_metrics = [m for m in metrics_to_compare if m in all_data]

    if not available_metrics:
        print("No comparison metrics found in data")
        return

    # Create subplot grid
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        runs_data = all_data[metric]

        for run_name, data in runs_data.items():
            steps = np.array(data['steps'])
            values = np.array(data['values'])

            if len(values) > 1:
                smoothed = exponential_moving_average(values, 0.7)
                ax.plot(steps, smoothed, label=run_name, linewidth=2)

        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(metric)
        ax.legend(loc='best', fontsize='x-small')
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    comparison_path = output_dir / "comparison_plot.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved comparison: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description='Export TensorBoard data as images')
    parser.add_argument('--logdir', type=str, default='logs/rsl_rl',
                        help='Path to TensorBoard logs directory')
    parser.add_argument('--output', type=str, default='tensorboard_exports',
                        help='Output directory for exported images')
    parser.add_argument('--smooth', type=float, default=0.6,
                        help='Smoothing factor (0-1, higher = smoother)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("TensorBoard Image Exporter")
    print(f"{'='*60}\n")

    # Load all data
    all_data = load_tensorboard_data(args.logdir)

    if not all_data:
        print("\nNo data to export!")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(all_data)} unique metrics")
    print(f"{'='*60}\n")

    # Export individual plots
    print("Exporting individual metric plots...")
    for tag, runs_data in all_data.items():
        plot_scalar(tag, runs_data, output_dir, smooth_factor=args.smooth)

    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(all_data, output_dir)

    print(f"\n{'='*60}")
    print(f"Export complete! Images saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
