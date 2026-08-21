# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
"""Plot RL vs PID velocity tracking comparison on same graph.

This script reads CSV data from both RL and PID controllers and generates
side-by-side comparison plots showing tracking performance differences.

Usage:
    # Auto-detect latest files from test/logs/ (RECOMMENDED)
    python scripts/ute/evobot/plotting/plot_comparison.py

    # Or specify files manually
    python scripts/ute/evobot/plotting/plot_comparison.py \
        source/isaaclab_assets/isaaclab_assets/evobot/test/logs/demo_rl_tracking_20260122_073817.csv \
        source/isaaclab_assets/isaaclab_assets/evobot/test/logs/demo_pid_tracking_20260122_073817.csv

    # Save without display
    python scripts/ute/evobot/plotting/plot_comparison.py \
        --output comparison_plot.png --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(description="Plot RL vs PID velocity tracking comparison")
parser.add_argument(
    "rl_csv",
    type=str,
    nargs="?",
    default=None,
    help="Path to RL tracking CSV file (optional, auto-detects if not provided)",
)
parser.add_argument(
    "pid_csv",
    type=str,
    nargs="?",
    default=None,
    help="Path to PID tracking CSV file (optional, auto-detects if not provided)",
)
parser.add_argument("--output", type=str, default="rl_vs_pid_comparison.png", help="Output plot filename")
parser.add_argument("--no-show", action="store_true", help="Don't display plot (only save)")
parser.add_argument("--dpi", type=int, default=300, help="Plot DPI (resolution)")
parser.add_argument("--figsize", type=str, default="16,10", help="Figure size (width,height in inches)")
args = parser.parse_args()

# Auto-detect files if not provided
if args.rl_csv is None or args.pid_csv is None:
    # Look in test/logs/ directory
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / "logs"

    if not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}")
        print("[INFO] Please generate data first using generate_fake_data.py")
        exit(1)

    # Find all CSV files
    rl_files = sorted(logs_dir.glob("demo_rl_tracking_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    pid_files = sorted(logs_dir.glob("demo_pid_tracking_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)

    if len(rl_files) == 0 or len(pid_files) == 0:
        print(f"[ERROR] Could not find RL or PID CSV files in {logs_dir}")
        print(f"[INFO] Found {len(rl_files)} RL files and {len(pid_files)} PID files")
        print("[INFO] Please generate data first using generate_fake_data.py")
        exit(1)

    # Use most recent files
    args.rl_csv = str(rl_files[0])
    args.pid_csv = str(pid_files[0])

    print("[INFO] Auto-detected files:")
    print(f"  RL:  {rl_files[0].name}")
    print(f"  PID: {pid_files[0].name}")

# Parse figsize
figsize = tuple(map(float, args.figsize.split(",")))

# Check if files exist
rl_path = Path(args.rl_csv)
pid_path = Path(args.pid_csv)

if not rl_path.exists():
    print(f"[ERROR] RL file not found: {rl_path}")
    exit(1)

if not pid_path.exists():
    print(f"[ERROR] PID file not found: {pid_path}")
    exit(1)

print(f"[INFO] Loading RL data from: {rl_path}")
print(f"[INFO] Loading PID data from: {pid_path}")

# Load data
df_rl = pd.read_csv(rl_path)
df_pid = pd.read_csv(pid_path)

print(f"[INFO] RL: {len(df_rl)} data points, time range: {df_rl['time'].min():.2f}s to {df_rl['time'].max():.2f}s")
print(f"[INFO] PID: {len(df_pid)} data points, time range: {df_pid['time'].min():.2f}s to {df_pid['time'].max():.2f}s")

# Create figure with 2 rows x 2 columns
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Color scheme
rl_color = "#2E86AB"  # Blue for RL
pid_color = "#A23B72"  # Purple for PID
cmd_color = "#F18F01"  # Orange for command
rl_error_color = "#06A77D"  # Green for RL error
pid_error_color = "#D81159"  # Red for PID error

# ============================================================================
# SUBPLOT 1 (Top-Left): Linear Velocity Tracking - RL vs PID
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Command line (shared)
ax1.plot(df_rl["time"], df_rl["vx_cmd"], "--", color=cmd_color, linewidth=2.5, label="Command", alpha=0.8, zorder=2)

# RL tracking
ax1.plot(df_rl["time"], df_rl["vx_actual"], "-", color=rl_color, linewidth=2.0, label="RL", alpha=0.85, zorder=4)

# PID tracking
ax1.plot(df_pid["time"], df_pid["vx_actual"], "-", color=pid_color, linewidth=2.0, label="PID", alpha=0.85, zorder=3)

# Tolerance band
ax1.fill_between(
    df_rl["time"],
    df_rl["vx_cmd"] * 0.95,
    df_rl["vx_cmd"] * 1.05,
    alpha=0.12,
    color=cmd_color,
    label="±5% tolerance",
    zorder=1,
)

ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)
ax1.set_ylabel("Linear Velocity (m/s)", fontsize=11, fontweight="bold")
ax1.set_title("Linear Velocity Tracking (vx)", fontsize=13, fontweight="bold", pad=10)
ax1.legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=2)
ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax1.set_xlim(df_rl["time"].min(), df_rl["time"].max())

# ============================================================================
# SUBPLOT 2 (Top-Right): Angular Velocity Tracking - RL vs PID
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Command line (shared)
ax2.plot(df_rl["time"], df_rl["wz_cmd"], "--", color=cmd_color, linewidth=2.5, label="Command", alpha=0.8, zorder=2)

# RL tracking
ax2.plot(df_rl["time"], df_rl["wz_actual"], "-", color=rl_color, linewidth=2.0, label="RL", alpha=0.85, zorder=4)

# PID tracking
ax2.plot(df_pid["time"], df_pid["wz_actual"], "-", color=pid_color, linewidth=2.0, label="PID", alpha=0.85, zorder=3)

# Tolerance band
ax2.fill_between(
    df_rl["time"],
    df_rl["wz_cmd"] * 0.95,
    df_rl["wz_cmd"] * 1.05,
    alpha=0.12,
    color=cmd_color,
    label="±5% tolerance",
    zorder=1,
)

ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)
ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=11, fontweight="bold")
ax2.set_title("Angular Velocity Tracking (ωz)", fontsize=13, fontweight="bold", pad=10)
ax2.legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=2)
ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax2.set_xlim(df_rl["time"].min(), df_rl["time"].max())

# ============================================================================
# SUBPLOT 3 (Bottom-Left): Linear Velocity Error - RL vs PID
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

# RL error
ax3.plot(df_rl["time"], df_rl["vx_error"], "-", color=rl_color, linewidth=2.0, label="RL Error", alpha=0.85)

# PID error
ax3.plot(df_pid["time"], df_pid["vx_error"], "-", color=pid_color, linewidth=2.0, label="PID Error", alpha=0.85)

ax3.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
ax3.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Linear Velocity Error (m/s)", fontsize=11, fontweight="bold")
ax3.set_title("Linear Velocity Tracking Error", fontsize=13, fontweight="bold", pad=10)
ax3.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax3.set_xlim(df_rl["time"].min(), df_rl["time"].max())

# ============================================================================
# SUBPLOT 4 (Bottom-Right): Angular Velocity Error - RL vs PID
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# RL error
ax4.plot(df_rl["time"], df_rl["wz_error"], "-", color=rl_color, linewidth=2.0, label="RL Error", alpha=0.85)

# PID error
ax4.plot(df_pid["time"], df_pid["wz_error"], "-", color=pid_color, linewidth=2.0, label="PID Error", alpha=0.85)

ax4.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
ax4.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Angular Velocity Error (rad/s)", fontsize=11, fontweight="bold")
ax4.set_title("Angular Velocity Tracking Error", fontsize=13, fontweight="bold", pad=10)
ax4.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax4.set_xlim(df_rl["time"].min(), df_rl["time"].max())

# ============================================================================
# Calculate and display statistics
# ============================================================================

print("\n" + "=" * 80)
print("TRACKING STATISTICS COMPARISON")
print("=" * 80)

# RL statistics
rl_vx_rmse = np.sqrt(np.mean(df_rl["vx_error"] ** 2))
rl_vx_mae = np.mean(np.abs(df_rl["vx_error"]))
rl_vx_max = np.max(np.abs(df_rl["vx_error"]))

rl_wz_rmse = np.sqrt(np.mean(df_rl["wz_error"] ** 2))
rl_wz_mae = np.mean(np.abs(df_rl["wz_error"]))
rl_wz_max = np.max(np.abs(df_rl["wz_error"]))

# PID statistics
pid_vx_rmse = np.sqrt(np.mean(df_pid["vx_error"] ** 2))
pid_vx_mae = np.mean(np.abs(df_pid["vx_error"]))
pid_vx_max = np.max(np.abs(df_pid["vx_error"]))

pid_wz_rmse = np.sqrt(np.mean(df_pid["wz_error"] ** 2))
pid_wz_mae = np.mean(np.abs(df_pid["wz_error"]))
pid_wz_max = np.max(np.abs(df_pid["wz_error"]))

print(f"\n{'Metric':<25} {'RL':>12} {'PID':>12} {'Improvement':>12}")
print("-" * 80)
print(f"{'Linear Velocity (vx):'}")
print(f"  {'RMSE (m/s)':<23} {rl_vx_rmse:>12.6f} {pid_vx_rmse:>12.6f} {(1 - rl_vx_rmse / pid_vx_rmse) * 100:>11.1f}%")
print(f"  {'MAE (m/s)':<23} {rl_vx_mae:>12.6f} {pid_vx_mae:>12.6f} {(1 - rl_vx_mae / pid_vx_mae) * 100:>11.1f}%")
print(f"  {'Max Error (m/s)':<23} {rl_vx_max:>12.6f} {pid_vx_max:>12.6f} {(1 - rl_vx_max / pid_vx_max) * 100:>11.1f}%")

print(f"\n{'Angular Velocity (wz):'}")
print(f"  {'RMSE (rad/s)':<23} {rl_wz_rmse:>12.6f} {pid_wz_rmse:>12.6f} {(1 - rl_wz_rmse / pid_wz_rmse) * 100:>11.1f}%")
print(f"  {'MAE (rad/s)':<23} {rl_wz_mae:>12.6f} {pid_wz_mae:>12.6f} {(1 - rl_wz_mae / pid_wz_mae) * 100:>11.1f}%")
print(
    f"  {'Max Error (rad/s)':<23} {rl_wz_max:>12.6f} {pid_wz_max:>12.6f} {(1 - rl_wz_max / pid_wz_max) * 100:>11.1f}%"
)

print("=" * 80)

# Add overall title
fig.suptitle("RL vs PID Velocity Tracking Comparison", fontsize=16, fontweight="bold", y=0.995)

# Add statistics comparison text box to top-left plot
stats_comparison = "Performance Comparison:\n"
stats_comparison += f"Linear: RL RMSE={rl_vx_rmse:.4f} vs PID={pid_vx_rmse:.4f} m/s\n"
stats_comparison += f"Angular: RL RMSE={rl_wz_rmse:.4f} vs PID={pid_wz_rmse:.4f} rad/s\n"
stats_comparison += f"RL Improvement: {(1 - rl_vx_rmse / pid_vx_rmse) * 100:.1f}% (vx), {(1 - rl_wz_rmse / pid_wz_rmse) * 100:.1f}% (ωz)"  # noqa: E501

ax1.text(
    0.02,
    0.02,
    stats_comparison,
    transform=ax1.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    zorder=10,
)

# Save plot
plt.tight_layout()
output_path = Path(args.output)
plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
print(f"\n✓ Comparison plot saved to: {output_path.absolute()}")

# Show plot
if not args.no_show:
    print("[INFO] Displaying plot (close window to exit)...")
    plt.show()
else:
    print("[INFO] Plot saved (use without --no-show to display)")

print("[INFO] Done!")
