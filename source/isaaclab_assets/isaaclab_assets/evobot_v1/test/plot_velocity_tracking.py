#!/usr/bin/env python3
"""Plot velocity tracking response from CSV data.

This script reads CSV data from test_rl_velocity_tracking.py and generates
professional plots with tracking performance analysis.

Usage:
    python plot_velocity_tracking.py logs/rl_velocity_tracking_20260116_123456.csv
    python plot_velocity_tracking.py logs/rl_velocity_tracking_20260116_123456.csv --output my_plot.png
    python plot_velocity_tracking.py logs/rl_velocity_tracking_20260116_123456.csv --no-show
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Plot velocity tracking response")
parser.add_argument("csv_file", type=str, help="Path to CSV file from test_rl_velocity_tracking.py")
parser.add_argument("--output", type=str, default="velocity_tracking_plot.png", help="Output plot filename")
parser.add_argument("--no-show", action="store_true", help="Don't display plot (only save)")
parser.add_argument("--dpi", type=int, default=300, help="Plot DPI (resolution)")
parser.add_argument("--figsize", type=str, default="14,10", help="Figure size (width,height in inches)")
args = parser.parse_args()

# Parse figsize
figsize = tuple(map(float, args.figsize.split(",")))

# Check if file exists
csv_path = Path(args.csv_file)
if not csv_path.exists():
    print(f"[ERROR] File not found: {csv_path}")
    exit(1)

print(f"[INFO] Loading data from: {csv_path}")

# Load data
df = pd.read_csv(csv_path)

print(f"[INFO] Loaded {len(df)} data points")
print(f"[INFO] Time range: {df['time'].min():.2f}s to {df['time'].max():.2f}s")

# Create figure with 3 subplots
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(3, 1, hspace=0.3)

# Subplot 1: Linear velocity tracking
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df["time"], df["vx_cmd"], "r--", linewidth=2.5, label="Command", alpha=0.8, zorder=2)
ax1.plot(df["time"], df["vx_actual"], "b-", linewidth=1.5, label="Actual", zorder=3)

# Add tolerance band
ax1.fill_between(
    df["time"],
    df["vx_cmd"] * 0.95,
    df["vx_cmd"] * 1.05,
    alpha=0.12,
    color="red",
    label="±5% tolerance",
    zorder=1,
)

# Add zero line
ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)

# Styling
ax1.set_ylabel("Linear Velocity (m/s)", fontsize=12, fontweight="bold")
ax1.set_title("RL Policy Velocity Tracking Performance", fontsize=14, fontweight="bold", pad=15)
ax1.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax1.set_xlim(df["time"].min(), df["time"].max())

# Subplot 2: Angular velocity tracking
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df["time"], df["wz_cmd"], "r--", linewidth=2.5, label="Command", alpha=0.8, zorder=2)
ax2.plot(df["time"], df["wz_actual"], "b-", linewidth=1.5, label="Actual", zorder=3)

# Add tolerance band
ax2.fill_between(
    df["time"],
    df["wz_cmd"] * 0.95,
    df["wz_cmd"] * 1.05,
    alpha=0.12,
    color="red",
    label="±5% tolerance",
    zorder=1,
)

# Add zero line
ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3, zorder=0)

# Styling
ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax2.set_xlim(df["time"].min(), df["time"].max())

# Subplot 3: Tracking errors
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(df["time"], df["vx_error"], "g-", linewidth=1.5, label="Linear Error", alpha=0.8)
ax3.plot(df["time"], df["wz_error"], "m-", linewidth=1.5, label="Angular Error", alpha=0.8)

# Add zero line
ax3.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

# Styling
ax3.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
ax3.set_ylabel("Tracking Error", fontsize=12, fontweight="bold")
ax3.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax3.set_xlim(df["time"].min(), df["time"].max())

# Calculate and display statistics
print("\n" + "=" * 80)
print("TRACKING STATISTICS")
print("=" * 80)

vx_rmse = np.sqrt(np.mean(df["vx_error"] ** 2))
vx_mae = np.mean(np.abs(df["vx_error"]))
vx_max_error = np.max(np.abs(df["vx_error"]))

wz_rmse = np.sqrt(np.mean(df["wz_error"] ** 2))
wz_mae = np.mean(np.abs(df["wz_error"]))
wz_max_error = np.max(np.abs(df["wz_error"]))

print(f"\nLinear Velocity (vx):")
print(f"  RMSE (Root Mean Square Error): {vx_rmse:.6f} m/s")
print(f"  MAE (Mean Absolute Error):     {vx_mae:.6f} m/s")
print(f"  Max Error:                     {vx_max_error:.6f} m/s")

print(f"\nAngular Velocity (wz):")
print(f"  RMSE (Root Mean Square Error): {wz_rmse:.6f} rad/s")
print(f"  MAE (Mean Absolute Error):     {wz_mae:.6f} rad/s")
print(f"  Max Error:                     {wz_max_error:.6f} rad/s")

# Add statistics text box to plot
stats_text = f"Linear: RMSE={vx_rmse:.4f} m/s, MAE={vx_mae:.4f} m/s\n"
stats_text += f"Angular: RMSE={wz_rmse:.4f} rad/s, MAE={wz_mae:.4f} rad/s"

ax1.text(
    0.02,
    0.02,
    stats_text,
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

print("=" * 80)

# Save plot
plt.tight_layout()
output_path = Path(args.output)
plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
print(f"\n✓ Plot saved to: {output_path.absolute()}")

# Show plot
if not args.no_show:
    print("[INFO] Displaying plot (close window to exit)...")
    plt.show()
else:
    print("[INFO] Plot saved (use without --no-show to display)")

print("[INFO] Done!")
