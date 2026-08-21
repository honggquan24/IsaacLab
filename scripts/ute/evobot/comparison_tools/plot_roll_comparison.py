# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
"""Plot comparison between PID and RL roll angle tracking.

Usage:
    python plot_roll_comparison.py \
        /path/to/pid_roll_tracking.csv \
        /path/to/rl_roll_tracking.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(description="Compare PID vs RL roll tracking")
parser.add_argument("pid_file", type=str, help="Path to PID roll tracking CSV file")
parser.add_argument("rl_file", type=str, help="Path to RL roll tracking CSV file")
parser.add_argument("--output", type=str, default="roll_comparison.png", help="Output plot filename")
args = parser.parse_args()

# Load data
print(f"\n{'=' * 80}")
print("LOADING DATA")
print(f"{'=' * 80}")
print(f"PID file: {args.pid_file}")
print(f"RL file:  {args.rl_file}")

df_pid = pd.read_csv(args.pid_file)
df_rl = pd.read_csv(args.rl_file)

# Convert to degrees
df_pid["roll_cmd_deg"] = np.rad2deg(df_pid["roll_cmd"])
df_pid["roll_actual_deg"] = np.rad2deg(df_pid["roll_actual"])
df_pid["roll_error_deg"] = np.rad2deg(df_pid["roll_error"])
df_pid["roll_abs_error_deg"] = np.abs(df_pid["roll_error_deg"])

df_rl["roll_cmd_deg"] = np.rad2deg(df_rl["roll_cmd"])
df_rl["roll_actual_deg"] = np.rad2deg(df_rl["roll_actual"])
df_rl["roll_error_deg"] = np.rad2deg(df_rl["roll_error"])
df_rl["roll_abs_error_deg"] = np.abs(df_rl["roll_error_deg"])

print(f"✓ PID data: {len(df_pid)} points")
print(f"✓ RL data:  {len(df_rl)} points")

# Compute statistics
print(f"\n{'=' * 80}")
print("STATISTICS")
print(f"{'=' * 80}")

pid_mae = df_pid["roll_abs_error_deg"].mean()
pid_max = df_pid["roll_abs_error_deg"].max()
pid_std = df_pid["roll_abs_error_deg"].std()

rl_mae = df_rl["roll_abs_error_deg"].mean()
rl_max = df_rl["roll_abs_error_deg"].max()
rl_std = df_rl["roll_abs_error_deg"].std()

print("\nPID Controller:")
print(f"  MAE (Mean Absolute Error): {pid_mae:.3f}°")
print(f"  Max Absolute Error:        {pid_max:.3f}°")
print(f"  Std Dev:                   {pid_std:.3f}°")

print("\nRL Policy:")
print(f"  MAE (Mean Absolute Error): {rl_mae:.3f}°")
print(f"  Max Absolute Error:        {rl_max:.3f}°")
print(f"  Std Dev:                   {rl_std:.3f}°")

improvement = ((pid_mae - rl_mae) / pid_mae) * 100 if pid_mae > 0 else 0
print("\nImprovement:")
print(f"  RL is {abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'} than PID in MAE")

# Create figure with 1x2 subplots for comparison (cleaner view)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("PID vs RL Roll Angle Tracking Comparison", fontsize=18, fontweight="bold", y=0.98)

# Plot 1: Absolute error comparison (MOST IMPORTANT for balance quality)
axes[0].plot(df_pid["time"], df_pid["roll_abs_error_deg"], "r-", linewidth=2, label="PID", alpha=0.8)
axes[0].plot(df_rl["time"], df_rl["roll_abs_error_deg"], "b-", linewidth=2.5, label="RL", alpha=0.9)
axes[0].axhline(y=0, color="k", linestyle="--", linewidth=1)
axes[0].set_xlabel("Time (s)", fontsize=13)
axes[0].set_ylabel("Absolute Error (deg)", fontsize=13)
axes[0].set_title("Absolute Error (Balance Quality)", fontsize=14, fontweight="bold")
axes[0].legend(loc="upper right", fontsize=11, framealpha=0.9)
axes[0].grid(True, alpha=0.3, linestyle="--")

# Add text annotation with statistics (no improvement %)
stats_text = f"PID MAE: {pid_mae:.2f}°\nRL MAE: {rl_mae:.2f}°"
axes[0].text(
    0.02,
    0.98,
    stats_text,
    transform=axes[0].transAxes,
    fontsize=11,
    verticalalignment="top",
    weight="bold",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8, edgecolor="navy", linewidth=2),
)

# Plot 2: Signed error comparison
axes[1].plot(df_pid["time"], df_pid["roll_error_deg"], "r-", linewidth=2, label="PID", alpha=0.8)
axes[1].plot(df_rl["time"], df_rl["roll_error_deg"], "b-", linewidth=2.5, label="RL", alpha=0.9)
axes[1].axhline(y=0, color="k", linestyle="-", linewidth=1.5)
axes[1].set_xlabel("Time (s)", fontsize=13)
axes[1].set_ylabel("Error (deg)", fontsize=13)
axes[1].set_title("Signed Error", fontsize=14, fontweight="bold")
axes[1].legend(loc="upper right", fontsize=11, framealpha=0.9)
axes[1].grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()

# Save plot
output_path = Path(args.output)
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"\n{'=' * 80}")
print(f"✓ Plot saved to: {output_path.absolute()}")
print(f"{'=' * 80}\n")

plt.show()
