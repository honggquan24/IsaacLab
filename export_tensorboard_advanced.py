#!/usr/bin/env python3
"""
Advanced TensorBoard image exporter with filtering options.
Usage examples:
  # Export all runs
  python export_tensorboard_advanced.py

  # Export specific task
  python export_tensorboard_advanced.py --task evobot_v1_velocity

  # Export specific run
  python export_tensorboard_advanced.py --run 2026-01-15_17-04-57

  # Export with custom metrics
  python export_tensorboard_advanced.py --metrics "Train/mean_reward,Loss/value_function"
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(logdir, task_filter=None, run_filter=None):
    """Load scalar data from TensorBoard with optional filtering."""
    print(f"Loading from: {logdir}")

    event_files = list(Path(logdir).rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"No event files found in {logdir}")
        return {}

    # Apply filters
    if task_filter:
        event_files = [f for f in event_files if task_filter in f.parent.parent.name]
        print(f"Filtering by task: {task_filter}")

    if run_filter:
        event_files = [f for f in event_files if run_filter in f.parent.name]
        print(f"Filtering by run: {run_filter}")

    print(f"Processing {len(event_files)} event file(s)")

    all_data = defaultdict(lambda: defaultdict(list))
    run_metadata = {}

    for event_file in event_files:
        run_name = event_file.parent.name
        task_name = event_file.parent.parent.name
        full_run_name = f"{task_name}/{run_name}"

        print(f"  Loading: {full_run_name}")

        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()

            scalar_tags = ea.Tags()['scalars']

            # Store metadata
            run_metadata[full_run_name] = {
                'task': task_name,
                'run': run_name,
                'num_metrics': len(scalar_tags)
            }

            for tag in scalar_tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                all_data[tag][full_run_name] = {
                    'steps': steps,
                    'values': values
                }

        except Exception as e:
            print(f"    Error: {e}")
            continue

    return all_data, run_metadata


def exponential_moving_average(values, smooth_factor=0.6):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]

    for value in values:
        smoothed_val = last * smooth_factor + (1 - smooth_factor) * value
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def plot_single_metric(tag, runs_data, output_dir, smooth_factor=0.6):
    """Plot a single metric across runs."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))

    for idx, (run_name, data) in enumerate(runs_data.items()):
        steps = np.array(data['steps'])
        values = np.array(data['values'])

        color = colors[idx]

        # Raw data (transparent)
        ax.plot(steps, values, alpha=0.2, linewidth=0.8, color=color)

        # Smoothed data
        if len(values) > 1:
            smoothed = exponential_moving_average(values, smooth_factor)
            ax.plot(steps, smoothed, label=run_name, linewidth=2.5, color=color)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(tag, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Save
    safe_tag = tag.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f"{safe_tag}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return output_path


def create_summary_report(run_metadata, all_data, output_dir):
    """Create a text summary report."""
    report_path = output_dir / "export_summary.txt"

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TensorBoard Export Summary\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Runs: {len(run_metadata)}\n")
        f.write(f"Total Metrics: {len(all_data)}\n\n")

        f.write("Runs:\n")
        f.write("-"*80 + "\n")
        for run_name, metadata in sorted(run_metadata.items()):
            f.write(f"  {run_name}\n")
            f.write(f"    Task: {metadata['task']}\n")
            f.write(f"    Metrics: {metadata['num_metrics']}\n\n")

        f.write("\nMetrics:\n")
        f.write("-"*80 + "\n")
        for tag in sorted(all_data.keys()):
            f.write(f"  {tag} ({len(all_data[tag])} runs)\n")

    print(f"  Summary: {report_path}")
    return report_path


def create_grid_comparison(all_data, output_dir, metrics_list=None):
    """Create grid comparison of key metrics."""

    if metrics_list is None:
        # Auto-select important metrics
        priority_keywords = ['reward', 'loss', 'mean_reward', 'episode_length']
        metrics_list = []

        for tag in all_data.keys():
            for keyword in priority_keywords:
                if keyword.lower() in tag.lower():
                    metrics_list.append(tag)
                    break

        metrics_list = metrics_list[:9]  # Limit to 9 for 3x3 grid

    if not metrics_list:
        print("No metrics to compare")
        return

    # Create grid
    n_metrics = len(metrics_list)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        runs_data = all_data[metric]

        colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))

        for color_idx, (run_name, data) in enumerate(runs_data.items()):
            steps = np.array(data['steps'])
            values = np.array(data['values'])

            if len(values) > 1:
                smoothed = exponential_moving_average(values, 0.7)
                ax.plot(steps, smoothed, label=run_name.split('/')[0],
                        linewidth=2, color=colors[color_idx])

        ax.set_xlabel('Steps', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    grid_path = output_dir / "metrics_grid.png"
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"  Grid: {grid_path}")
    return grid_path


def main():
    parser = argparse.ArgumentParser(
        description='Advanced TensorBoard image exporter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Export all runs:
    python export_tensorboard_advanced.py

  Export specific task:
    python export_tensorboard_advanced.py --task evobot_v1_velocity

  Export specific run:
    python export_tensorboard_advanced.py --run 2026-01-15_17-04-57

  Custom metrics:
    python export_tensorboard_advanced.py --metrics "Train/mean_reward,Loss/value_function"
        """
    )

    parser.add_argument('--logdir', type=str, default='logs/rsl_rl',
                        help='TensorBoard logs directory')
    parser.add_argument('--output', type=str, default='tensorboard_exports',
                        help='Output directory')
    parser.add_argument('--task', type=str, default=None,
                        help='Filter by task name (e.g., evobot_v1_velocity)')
    parser.add_argument('--run', type=str, default=None,
                        help='Filter by run timestamp (e.g., 2026-01-15_17-04-57)')
    parser.add_argument('--metrics', type=str, default=None,
                        help='Comma-separated list of metrics to export')
    parser.add_argument('--smooth', type=float, default=0.6,
                        help='Smoothing factor (0-1)')
    parser.add_argument('--no-grid', action='store_true',
                        help='Skip grid comparison plot')

    args = parser.parse_args()

    # Create output
    output_dir = Path(args.output)
    if args.task:
        output_dir = output_dir / args.task
    if args.run:
        output_dir = output_dir / args.run
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("TensorBoard Advanced Exporter")
    print("="*80 + "\n")

    # Load data
    all_data, run_metadata = load_tensorboard_data(
        args.logdir,
        task_filter=args.task,
        run_filter=args.run
    )

    if not all_data:
        print("\nNo data to export!")
        return

    print(f"\n{'='*80}")
    print(f"Loaded {len(all_data)} metrics from {len(run_metadata)} runs")
    print(f"{'='*80}\n")

    # Filter metrics if specified
    if args.metrics:
        metrics_to_export = [m.strip() for m in args.metrics.split(',')]
        all_data = {k: v for k, v in all_data.items() if k in metrics_to_export}
        print(f"Exporting {len(all_data)} specified metrics\n")

    # Export plots
    print("Exporting plots...")
    for tag, runs_data in all_data.items():
        output_path = plot_single_metric(tag, runs_data, output_dir, args.smooth)
        print(f"  ✓ {tag}")

    # Create grid comparison
    if not args.no_grid:
        print("\nCreating grid comparison...")
        create_grid_comparison(all_data, output_dir)

    # Create summary
    print("\nGenerating summary...")
    create_summary_report(run_metadata, all_data, output_dir)

    print(f"\n{'='*80}")
    print(f"✓ Export complete!")
    print(f"  Output: {output_dir.absolute()}")
    print(f"  Total images: {len(all_data) + (0 if args.no_grid else 1)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
