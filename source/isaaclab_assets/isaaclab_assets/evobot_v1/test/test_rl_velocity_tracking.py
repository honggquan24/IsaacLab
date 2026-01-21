#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test RL policy velocity tracking performance with step response analysis.

This script tests how well the trained RL policy tracks velocity commands.
It generates step commands and logs both commanded and actual velocities.
Output CSV can be used to plot tracking response curves.

Usage:
    # Test with trained RL policy
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_rl_velocity_tracking.py \
        --load_run 2026-01-15_01-19-39 \
        --checkpoint model_500.pt

    # Custom test sequence (step values)
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/test_rl_velocity_tracking.py \
        --load_run 2026-01-21_08-24-47 \
        --checkpoint model_1410.pt \
        --step_duration 10.0 \
        --step_values "0.0,0.5,0.0,-0.3,0.0,0.8,0.0"

    # Test only linear velocity
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_rl_velocity_tracking.py \
        --load_run 2026-01-15_01-19-39 \
        --checkpoint model_500.pt \
        --test_linear

    # Test only angular velocity
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_rl_velocity_tracking.py \
        --load_run 2026-01-15_01-19-39 \
        --checkpoint model_500.pt \
        --test_angular

Output:
    - Console: Real-time tracking data
    - CSV: logs/rl_velocity_tracking_<timestamp>.csv
    - Auto-generated Python plotting script

CSV Columns:
    time, vx_cmd, vx_actual, wz_cmd, wz_actual, vx_error, wz_error
"""

import argparse
import torch
import os
import csv
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="RL Policy Velocity Tracking Test")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Velocity-Play", help="Task name")

# RL Policy loading (REQUIRED)
parser.add_argument("--load_run", type=str, required=True, help="Run directory name (e.g., 2026-01-15_01-19-39)")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")

# Test sequence parameters
parser.add_argument("--step_duration", type=float, default=10.0, help="Duration of each step command (seconds)")
parser.add_argument(
    "--step_values",
    type=str,
    default="0.0,0.1,0.0,-0.1,0.0",
    help="Comma-separated velocity step values",
)
parser.add_argument("--test_linear", action="store_true", help="Test linear velocity (vx) - default if neither specified")
parser.add_argument("--test_angular", action="store_true", help="Test angular velocity (wz)")

# Logging parameters
parser.add_argument("--log_interval", type=int, default=1, help="Log every N steps (1=every step)")

parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


class TrackingLogger:
    """CSV logger for velocity tracking data."""

    def __init__(self, log_dir: str = "logs", filename_prefix: str = "rl_velocity_tracking"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")

        # Create CSV file with header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "vx_cmd", "vx_actual", "wz_cmd", "wz_actual", "vx_error", "wz_error"])

        print(f"[INFO] Logging to: {self.filepath}")

    def log(self, time: float, vx_cmd: float, vx_actual: float, wz_cmd: float, wz_actual: float):
        """Log a single timestep."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{time:.4f}",
                    f"{vx_cmd:.6f}",
                    f"{vx_actual:.6f}",
                    f"{wz_cmd:.6f}",
                    f"{wz_actual:.6f}",
                    f"{vx_cmd - vx_actual:.6f}",
                    f"{wz_cmd - wz_actual:.6f}",
                ]
            )

    def get_filepath(self) -> str:
        return self.filepath


def compute_step_metrics(time_data, cmd_data, actual_data, step_start_time, step_end_time):
    """Compute tracking metrics for a single step.

    Returns dict with: rise_time, settling_time, overshoot, steady_state_error
    """
    # Get data for this step
    mask = (time_data >= step_start_time) & (time_data < step_end_time)
    t = time_data[mask] - step_start_time
    cmd = cmd_data[mask]
    actual = actual_data[mask]

    if len(t) < 10:
        return None

    cmd_value = cmd[0]

    if abs(cmd_value) < 0.01:
        return None  # Skip zero commands

    # Rise time (10% to 90%)
    threshold_10 = abs(cmd_value) * 0.1
    threshold_90 = abs(cmd_value) * 0.9
    idx_10 = np.where(np.abs(actual) >= threshold_10)[0]
    idx_90 = np.where(np.abs(actual) >= threshold_90)[0]

    rise_time = None
    if len(idx_10) > 0 and len(idx_90) > 0:
        rise_time = t[idx_90[0]] - t[idx_10[0]]

    # Settling time (within 5%)
    threshold_settle = abs(cmd_value) * 0.05
    settled = np.abs(actual - cmd_value) <= threshold_settle

    settling_time = None
    if np.any(settled):
        for i in range(len(t) - 10):
            if np.all(settled[i : i + 10]):
                settling_time = t[i]
                break

    # Overshoot
    overshoot = 0.0
    max_val = np.max(np.abs(actual))
    if max_val > abs(cmd_value):
        overshoot = (max_val - abs(cmd_value)) / abs(cmd_value) * 100.0

    # Steady-state error (last 20% of data)
    steady_idx = int(len(actual) * 0.8)
    ss_error = np.mean(cmd[steady_idx:] - actual[steady_idx:])

    return {
        "command": cmd_value,
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot": overshoot,
        "steady_state_error": ss_error,
    }


def load_policy(env, agent_cfg, checkpoint_path: str):
    """Load trained RL policy."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading policy from: {checkpoint_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Policy loaded successfully")
    return policy


def main():
    """Main function."""

    # Parse step values
    step_values = [float(x) for x in args_cli.step_values.split(",")]

    # Default: test linear if neither specified
    test_linear = args_cli.test_linear or not args_cli.test_angular
    test_angular = args_cli.test_angular

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Set episode length
    total_duration = len(step_values) * args_cli.step_duration
    env_cfg.episode_length_s = total_duration + 5.0

    env = gym.make(args_cli.task, cfg=env_cfg)

    print("\n" + "=" * 80)
    print("RL POLICY VELOCITY TRACKING TEST")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Test duration: {total_duration:.1f}s ({len(step_values)} steps × {args_cli.step_duration:.1f}s)")
    print(f"Step values: {step_values}")
    print(f"Testing: {'Linear (vx)' if test_linear else ''} {'Angular (wz)' if test_angular else ''}")
    print("=" * 80)

    # Load agent config
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Wrap environment
    env_wrapped = RslRlVecEnvWrapper(env)

    # Load policy
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    policy = load_policy(env_wrapped, agent_cfg, checkpoint_path)

    # Initialize logger
    logger = TrackingLogger()

    # Reset environment
    env.reset()
    obs = env_wrapped.get_observations()

    # Get robot
    robot = env.unwrapped.scene["robot"]

    # Get timestep
    dt = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation
    control_freq = 1.0 / dt

    print(f"\n[INFO] Control frequency: {control_freq:.1f} Hz (dt={dt:.4f}s)")
    print("[INFO] Starting test...\n")

    # Data storage
    time_log = []
    vx_cmd_log = []
    vx_actual_log = []
    wz_cmd_log = []
    wz_actual_log = []
    step_times = [0.0]

    # Main loop
    step_count = 0
    elapsed_time = 0.0
    current_step_idx = 0

    # Get command manager (velocity commands are managed by environment)
    command_manager = env.unwrapped.command_manager

    print(f"{'Time':>8s} {'VxCmd':>10s} {'VxAct':>10s} {'WzCmd':>10s} {'WzAct':>10s} {'VxErr':>10s} {'WzErr':>10s}")
    print("-" * 78)

    try:
        while simulation_app.is_running() and elapsed_time < total_duration:
            with torch.inference_mode():
                # Update command manually based on step index
                if elapsed_time >= (current_step_idx + 1) * args_cli.step_duration:
                    current_step_idx += 1
                    if current_step_idx < len(step_values):
                        step_times.append(elapsed_time)
                        print(f"\n[STEP {current_step_idx}] New target: {step_values[current_step_idx]:.2f}")
                        print(f"{'Time':>8s} {'VxCmd':>10s} {'VxAct':>10s} {'WzCmd':>10s} {'WzAct':>10s} {'VxErr':>10s} {'WzErr':>10s}")
                        print("-" * 78)

                # Set command manually (override environment's command)
                # Access via command manager's active terms
                if len(command_manager.active_terms) > 0:
                    term_name = command_manager.active_terms[0]
                    term_command = command_manager.get_command(term_name)

                    if test_linear:
                        term_command[:, 0] = step_values[current_step_idx]  # vx
                    else:
                        term_command[:, 0] = 0.0

                    if test_angular:
                        term_command[:, 1] = step_values[current_step_idx]  # wz
                    else:
                        term_command[:, 1] = 0.0

                # Get current velocity
                vel_current = torch.stack(
                    [
                        robot.data.root_lin_vel_b[:, 0],  # vx
                        robot.data.root_ang_vel_b[:, 2],  # wz
                    ],
                    dim=-1,
                )

                # Get action from policy
                actions = policy(obs)

                # Step environment
                obs, reward, dones, _ = env_wrapped.step(actions)

                # Log data
                if step_count % args_cli.log_interval == 0:
                    # Get current command
                    if len(command_manager.active_terms) > 0:
                        term_name = command_manager.active_terms[0]
                        cmd_tensor = command_manager.get_command(term_name)
                        vx_cmd = cmd_tensor[0, 0].item()
                        wz_cmd = cmd_tensor[0, 1].item()
                    else:
                        vx_cmd = 0.0
                        wz_cmd = 0.0
                    vx_actual = vel_current[0, 0].item()
                    wz_actual = vel_current[0, 1].item()

                    time_log.append(elapsed_time)
                    vx_cmd_log.append(vx_cmd)
                    vx_actual_log.append(vx_actual)
                    wz_cmd_log.append(wz_cmd)
                    wz_actual_log.append(wz_actual)

                    logger.log(elapsed_time, vx_cmd, vx_actual, wz_cmd, wz_actual)

                    # Print every 50 steps
                    if step_count % 50 == 0:
                        vx_err = vx_cmd - vx_actual
                        wz_err = wz_cmd - wz_actual
                        print(
                            f"{elapsed_time:8.2f} {vx_cmd:10.3f} {vx_actual:10.3f} {wz_cmd:10.3f} {wz_actual:10.3f} {vx_err:10.3f} {wz_err:10.3f}"
                        )

                step_count += 1
                elapsed_time += dt

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Compute metrics
    print("\n" + "=" * 80)
    print("TRACKING PERFORMANCE METRICS")
    print("=" * 80)

    time_array = np.array(time_log)
    vx_cmd_array = np.array(vx_cmd_log)
    vx_actual_array = np.array(vx_actual_log)
    wz_cmd_array = np.array(wz_cmd_log)
    wz_actual_array = np.array(wz_actual_log)

    if test_linear:
        print("\nLinear Velocity (vx) Tracking:")
        for i in range(len(step_times) - 1):
            metrics = compute_step_metrics(time_array, vx_cmd_array, vx_actual_array, step_times[i], step_times[i + 1])
            if metrics:
                print(f"  Step {i}: cmd={metrics['command']:+.3f} m/s")
                if metrics["rise_time"]:
                    print(f"    Rise time: {metrics['rise_time']:.3f}s")
                if metrics["settling_time"]:
                    print(f"    Settling time: {metrics['settling_time']:.3f}s")
                print(f"    Overshoot: {metrics['overshoot']:.2f}%")
                print(f"    Steady-state error: {metrics['steady_state_error']:+.4f} m/s")

    if test_angular:
        print("\nAngular Velocity (wz) Tracking:")
        for i in range(len(step_times) - 1):
            metrics = compute_step_metrics(time_array, wz_cmd_array, wz_actual_array, step_times[i], step_times[i + 1])
            if metrics:
                print(f"  Step {i}: cmd={metrics['command']:+.3f} rad/s")
                if metrics["rise_time"]:
                    print(f"    Rise time: {metrics['rise_time']:.3f}s")
                if metrics["settling_time"]:
                    print(f"    Settling time: {metrics['settling_time']:.3f}s")
                print(f"    Overshoot: {metrics['overshoot']:.2f}%")
                print(f"    Steady-state error: {metrics['steady_state_error']:+.4f} rad/s")

    print("=" * 80)

    # Generate plotting code
    print("\n" + "=" * 80)
    print("PLOT TRACKING RESPONSE:")
    print("=" * 80)
    print(f"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('{logger.get_filepath()}')

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot 1: Linear velocity tracking
axes[0].plot(df['time'], df['vx_cmd'], 'r--', linewidth=2.5, label='Command', alpha=0.8)
axes[0].plot(df['time'], df['vx_actual'], 'b-', linewidth=1.5, label='Actual')
axes[0].fill_between(df['time'],
                      df['vx_cmd']*0.95, df['vx_cmd']*1.05,
                      alpha=0.15, color='red', label='±5% tolerance')
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[0].set_ylabel('Linear Velocity (m/s)', fontsize=12, fontweight='bold')
axes[0].set_title('RL Policy Velocity Tracking Performance', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=11)
axes[0].grid(True, alpha=0.3, linestyle='--')

# Plot 2: Angular velocity tracking
axes[1].plot(df['time'], df['wz_cmd'], 'r--', linewidth=2.5, label='Command', alpha=0.8)
axes[1].plot(df['time'], df['wz_actual'], 'b-', linewidth=1.5, label='Actual')
axes[1].fill_between(df['time'],
                      df['wz_cmd']*0.95, df['wz_cmd']*1.05,
                      alpha=0.15, color='red', label='±5% tolerance')
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=11)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('rl_velocity_tracking.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to: rl_velocity_tracking.png")
plt.show()
""")
    print("=" * 80)

    # Close
    print(f"\n[INFO] Data saved to: {logger.get_filepath()}")
    env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
