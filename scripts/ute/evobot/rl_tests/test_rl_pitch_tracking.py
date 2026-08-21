# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test RL policy orientation angle tracking performance with step response analysis.

This script tests how well the trained RL policy tracks orientation angle commands (roll/pitch/yaw).
It generates step commands for the selected axis and logs both commanded and actual values.
Output CSV can be used to plot tracking response curves and compare with PID controller.
Supports action smoothing to reduce oscillations.

Usage:
    # Test pitch angle tracking - DEFAULT
    ./isaaclab.sh -p scripts/ute/evobot/rl_tests/test_rl_pitch_tracking.py \
        --load_run 2026-01-23_02-20-53 \
        --checkpoint model_700.pt \
        --test_axis pitch \
        --step_duration 5.0 \
        --step_values "0.0,0.1,-0.1,0.0"

    # Test roll angle tracking
    ./isaaclab.sh -p scripts/ute/evobot/rl_tests/test_rl_pitch_tracking.py \
        --load_run 2026-01-23_02-20-53 \
        --checkpoint model_700.pt \
        --test_axis roll \
        --step_values "0.0,0.1,-0.1,0.0"

    # Test with arm finetune task
    ./isaaclab.sh -p scripts/ute/evobot/rl_tests/test_rl_pitch_tracking.py \
        --task Isaac-Evobot-Arm-FineTune \
        --load_run 2026-01-19_21-41-30 \
        --checkpoint model_1045.pt \
        --test_axis pitch \
        --step_duration 5.0 \
        --step_values "0.0,0.15,-0.15,0.0"

    # Test with action smoothing
    ./isaaclab.sh -p scripts/ute/evobot/rl_tests/test_rl_pitch_tracking.py \
        --load_run 2026-01-23_02-20-53 \
        --checkpoint model_700.pt \
        --step_values "0.0,0.2,-0.2,0.0" \
        --action_smoothing 0.5

    # Test with adaptive smoothing
    ./isaaclab.sh -p scripts/ute/evobot/rl_tests/test_rl_pitch_tracking.py \
        --load_run 2026-01-23_02-20-53 \
        --checkpoint model_700.pt \
        --step_values "0.0,0.15,-0.15,0.0" \
        --adaptive_smoothing \
        --smoothing_fast 0.3 \
        --smoothing_slow 0.7

Output:
    - Console: Real-time tracking data
    - CSV: logs/rl_pitch_tracking_<timestamp>.csv
    - Auto-generated Python plotting script

CSV Columns:
    time, angle_cmd, angle_actual, angle_error, height
    (where "angle" is the selected axis: roll, pitch, or yaw)

    Note: All angles are RELATIVE to initial baseline orientation.
    This means angle_actual=0.0 corresponds to robot's natural upright position,
    not absolute angle=0 in world frame.

Note:
    This script is designed to be compared with test_pid_balance_simple.py
    to evaluate RL policy vs PID controller for orientation control.

    The script measures RELATIVE angle changes from the robot's natural upright
    position (baseline), not absolute angles in world frame.
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="RL Policy Orientation Angle Tracking Test")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-Velocity-Play", help="Task name")

# RL Policy loading (REQUIRED)
parser.add_argument("--load_run", type=str, required=True, help="Run directory name (e.g., 2026-01-15_01-19-39)")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")

# Test axis selection
parser.add_argument(
    "--test_axis",
    type=str,
    default="pitch",
    choices=["roll", "pitch", "yaw"],
    help="Axis to test: roll, pitch, or yaw (default: pitch)",
)

# Test sequence parameters
parser.add_argument("--step_duration", type=float, default=10.0, help="Duration of each step command (seconds)")
parser.add_argument(
    "--step_values",
    type=str,
    default="0.0,0.1,0.0,-0.1,0.0",
    help="Comma-separated angle step values in radians",
)

# Logging parameters
parser.add_argument("--log_interval", type=int, default=1, help="Log every N steps (1=every step)")

# Action smoothing
parser.add_argument(
    "--action_smoothing",
    type=float,
    default=0.0,
    help="Action smoothing factor (0.0=no smoothing, 0.9=heavy smoothing)",
)
parser.add_argument(
    "--adaptive_smoothing",
    action="store_true",
    help="Enable adaptive smoothing (alpha adjusts based on command changes)",
)
parser.add_argument("--smoothing_fast", type=float, default=0.2, help="Alpha when command changes (fast response)")
parser.add_argument("--smoothing_slow", type=float, default=0.7, help="Alpha when command stable (heavy smoothing)")
parser.add_argument(
    "--command_change_threshold", type=float, default=0.05, help="Threshold to detect command change (radians)"
)

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
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


class AngleTrackingLogger:
    """CSV logger for orientation angle tracking data."""

    def __init__(self, axis_name: str, log_dir: str | None = None):
        # Default to test/logs/ directory relative to this script
        if log_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, "logs")

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"rl_{axis_name}_tracking"
        self.filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")
        self.axis_name = axis_name

        # Create CSV file with header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", f"{axis_name}_cmd", f"{axis_name}_actual", f"{axis_name}_error", "height"])

        print(f"[INFO] Logging to: {self.filepath}")

    def log(self, time: float, angle_cmd: float, angle_actual: float, height: float):
        """Log a single timestep."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{time:.4f}",
                    f"{angle_cmd:.6f}",
                    f"{angle_actual:.6f}",
                    f"{angle_cmd - angle_actual:.6f}",
                    f"{height:.6f}",
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


def get_angle_from_imu(imu_sensor, axis: str):
    """
    Get orientation angle from IMU quaternion
    Similar to test_pid_balance_simple.py logic

    Args:
        imu_sensor: IMU sensor object
        axis: "roll", "pitch", or "yaw"

    Returns:
        Angle tensor for the selected axis
    """
    # Get quaternion (world frame)
    quat = imu_sensor.data.quat_w

    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    # Convert to Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quat)

    # Select the requested axis
    if axis == "roll":
        angle = roll
    elif axis == "pitch":
        angle = pitch
    elif axis == "yaw":
        angle = yaw
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'roll', 'pitch', or 'yaw'")

    # Clamp angle to avoid extreme values
    angle = torch.clamp(angle, -torch.pi, torch.pi)

    return angle


def main():
    """Main function."""

    # Parse step values
    step_values = [float(x) for x in args_cli.step_values.split(",")]
    test_axis = args_cli.test_axis.lower()
    axis_name_upper = test_axis.capitalize()

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Set episode length
    total_duration = len(step_values) * args_cli.step_duration
    env_cfg.episode_length_s = total_duration + 5.0

    env = gym.make(args_cli.task, cfg=env_cfg)

    print("\n" + "=" * 80)
    print(f"RL POLICY {axis_name_upper.upper()} ANGLE TRACKING TEST (RELATIVE)")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Test axis: {axis_name_upper}")
    print(f"Test duration: {total_duration:.1f}s ({len(step_values)} steps × {args_cli.step_duration:.1f}s)")
    print(f"Step values (relative): {step_values} (radians)")
    print(f"Step values (relative): {[np.rad2deg(v) for v in step_values]} (degrees)")
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
    logger = AngleTrackingLogger(axis_name=test_axis)

    # Reset environment
    env.reset()
    obs = env_wrapped.get_observations()

    # Get robot and environment internals
    env_unwrapped = env.unwrapped  # type: ignore
    robot = env_unwrapped.scene["robot"]
    imu = env_unwrapped.scene["imu"]

    # Get timestep
    dt = env_unwrapped.physics_dt * env_unwrapped.cfg.decimation
    control_freq = 1.0 / dt

    # IMPORTANT: Get initial angle offset (baseline when robot is "upright")
    # We'll track relative changes from this baseline
    initial_angle = get_angle_from_imu(imu, test_axis)[0].item()
    print(f"\n[INFO] Initial {test_axis} (baseline): {initial_angle:.4f} rad ({np.rad2deg(initial_angle):.2f}°)")
    print(f"[INFO] All {test_axis} commands will be RELATIVE to this baseline")
    print(f"[INFO] Control frequency: {control_freq:.1f} Hz (dt={dt:.4f}s)")

    # Action smoothing setup
    alpha = args_cli.action_smoothing
    prev_actions = None
    prev_angle_command = None
    use_adaptive = args_cli.adaptive_smoothing

    if alpha > 0.0 or use_adaptive:
        if use_adaptive:
            print("[INFO] Adaptive EMA smoothing enabled:")
            print(f"      - Fast alpha (command change): {args_cli.smoothing_fast:.2f}")
            print(f"      - Slow alpha (stable): {args_cli.smoothing_slow:.2f}")
            print(f"      - Change threshold: {args_cli.command_change_threshold:.3f} rad")
        else:
            print(f"[INFO] Fixed EMA smoothing enabled: alpha={alpha:.2f}")
        print("      Formula: action_smooth = alpha * prev_action + (1-alpha) * current_action")

    print("[INFO] Starting test...\n")

    # Data storage
    time_log = []
    angle_cmd_log = []
    angle_actual_log = []
    height_log = []
    step_times = [0.0]

    # Main loop
    step_count = 0
    elapsed_time = 0.0
    current_step_idx = 0

    col_name = f"{axis_name_upper}Cmd"
    col_actual = f"{axis_name_upper}Act"
    print(f"{'Time':>8s} {col_name:>10s} {col_actual:>10s} {'Error':>10s} {'Height':>10s}")
    print(f"(All {test_axis} angles RELATIVE to baseline, in degrees)")
    print("-" * 68)

    try:
        while simulation_app.is_running() and elapsed_time < total_duration:
            with torch.inference_mode():
                # Check if we need to update to next step command
                if elapsed_time >= (current_step_idx + 1) * args_cli.step_duration:
                    current_step_idx += 1
                    if current_step_idx < len(step_values):
                        step_times.append(elapsed_time)
                        print(
                            f"\n[STEP {current_step_idx}] New target: {step_values[current_step_idx]:.3f} rad ({np.rad2deg(step_values[current_step_idx]):.2f}°)"  # noqa: E501
                        )
                        print(f"{'Time':>8s} {col_name:>10s} {col_actual:>10s} {'Error':>10s} {'Height':>10s}")
                        print("-" * 68)

                # IMPORTANT: For angle tracking, we don't have a command manager term
                # Instead, we'll use the current angle as a reference
                # The policy should maintain angle at target value

                # Get current angle from IMU
                angle_actual_absolute = get_angle_from_imu(imu, test_axis)
                angle_actual = angle_actual_absolute - initial_angle  # Relative to baseline

                # Command is relative to baseline
                angle_cmd_value = step_values[current_step_idx]

                # Update observations (policy will use its own internal state)
                obs = env_wrapped.get_observations()

                # Get action from policy using updated observations
                actions = policy(obs)

                # Apply action smoothing (exponential moving average)
                if alpha > 0.0 or use_adaptive:
                    if prev_actions is None:
                        # First step: initialize with current action
                        prev_actions = actions.clone()
                        if use_adaptive:
                            prev_angle_command = angle_cmd_value
                    else:
                        # Determine alpha (fixed or adaptive)
                        current_alpha = alpha

                        if use_adaptive:
                            # Adaptive smoothing: adjust alpha based on command changes
                            if prev_angle_command is not None:
                                # Detect command change
                                cmd_diff = abs(angle_cmd_value - prev_angle_command)

                                if cmd_diff > args_cli.command_change_threshold:
                                    # Command changed: use fast alpha (low value = fast response)
                                    current_alpha = args_cli.smoothing_fast
                                else:
                                    # Command stable: use slow alpha (high value = heavy smoothing)
                                    current_alpha = args_cli.smoothing_slow

                                prev_angle_command = angle_cmd_value
                            else:
                                current_alpha = args_cli.smoothing_slow

                        # Apply smoothing: action_smooth = alpha * prev_action + (1-alpha) * current_action
                        actions = current_alpha * prev_actions + (1.0 - current_alpha) * actions
                        prev_actions = actions.clone()

                # Get current height
                height_current = robot.data.root_pos_w[:, 2]

                # Step environment
                obs, reward, dones, _ = env_wrapped.step(actions)

                # Log data
                if step_count % args_cli.log_interval == 0:
                    angle_actual_value = angle_actual[0].item()  # Already relative to baseline
                    height_value = height_current[0].item()

                    time_log.append(elapsed_time)
                    angle_cmd_log.append(angle_cmd_value)
                    angle_actual_log.append(angle_actual_value)
                    height_log.append(height_value)

                    logger.log(elapsed_time, angle_cmd_value, angle_actual_value, height_value)

                    # Print every 50 steps
                    if step_count % 50 == 0:
                        angle_error = angle_cmd_value - angle_actual_value
                        print(
                            f"{elapsed_time:8.2f} {np.rad2deg(angle_cmd_value):10.2f} {np.rad2deg(angle_actual_value):10.2f} {np.rad2deg(angle_error):10.2f} {height_value:10.3f}"  # noqa: E501
                        )

                step_count += 1
                elapsed_time += dt

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Compute metrics
    print("\n" + "=" * 80)
    print(f"{axis_name_upper.upper()} ANGLE TRACKING PERFORMANCE METRICS")
    print("=" * 80)

    time_array = np.array(time_log)
    angle_cmd_array = np.array(angle_cmd_log)
    angle_actual_array = np.array(angle_actual_log)

    print(f"\n{axis_name_upper} Angle Tracking:")
    for i in range(len(step_times) - 1):
        metrics = compute_step_metrics(
            time_array, angle_cmd_array, angle_actual_array, step_times[i], step_times[i + 1]
        )
        if metrics:
            print(f"  Step {i}: cmd={metrics['command']:+.3f} rad ({np.rad2deg(metrics['command']):+.2f}°)")
            if metrics["rise_time"]:
                print(f"    Rise time: {metrics['rise_time']:.3f}s")
            if metrics["settling_time"]:
                print(f"    Settling time: {metrics['settling_time']:.3f}s")
            print(f"    Overshoot: {metrics['overshoot']:.2f}%")
            print(
                f"    Steady-state error: {metrics['steady_state_error']:+.4f} rad ({np.rad2deg(metrics['steady_state_error']):+.3f}°)"  # noqa: E501
            )

    print("=" * 80)

    # Generate plotting code
    print("\n" + "=" * 80)
    print(f"PLOT {axis_name_upper.upper()} TRACKING RESPONSE:")
    print("=" * 80)
    print(f"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('{logger.get_filepath()}')

# Convert to degrees
df['{test_axis}_cmd_deg'] = np.rad2deg(df['{test_axis}_cmd'])
df['{test_axis}_actual_deg'] = np.rad2deg(df['{test_axis}_actual'])
df['{test_axis}_error_deg'] = np.rad2deg(df['{test_axis}_error'])
df['{test_axis}_abs_error_deg'] = np.abs(df['{test_axis}_error_deg'])

# Create figure with 2x2 subplots (same as PID test)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('RL Policy - {axis_name_upper} Angle Tracking (Relative to Baseline)', fontsize=14, fontweight='bold')

# Plot 1: Absolute error (shows balance quality better than raw angle)
axes[0, 0].plot(df['time'], df['{test_axis}_abs_error_deg'], 'b-', linewidth=2)
axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[0, 0].axhline(y=2, color='g', linestyle='--', linewidth=1, alpha=0.5, label='±2° threshold')
axes[0, 0].fill_between(df['time'], 0, 2, alpha=0.1, color='green', label='Good balance zone')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Absolute Error (deg)')
axes[0, 0].set_title('{axis_name_upper} Absolute Error (Balance Quality)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: {axis_name_upper} error (signed)
axes[0, 1].plot(df['time'], df['{test_axis}_error_deg'], 'r-', linewidth=2)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[0, 1].fill_between(df['time'], -2, 2, alpha=0.1, color='green', label='±2° tolerance')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Error (deg)')
axes[0, 1].set_title('{axis_name_upper} Error (Signed)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: {axis_name_upper} command vs actual (overlay)
axes[1, 0].plot(df['time'], df['{test_axis}_cmd_deg'], 'r--', linewidth=2, label='Command', alpha=0.7)
axes[1, 0].plot(df['time'], df['{test_axis}_actual_deg'], 'b-', linewidth=2, label='Actual')
axes[1, 0].fill_between(df['time'],
                        df['{test_axis}_cmd_deg']*0.95, df['{test_axis}_cmd_deg']*1.05,
                        alpha=0.15, color='red', label='±5% tolerance')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('{axis_name_upper} Angle (deg)')
axes[1, 0].set_title('Command vs Actual')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Robot height
axes[1, 1].plot(df['time'], df['height'], 'm-', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Height (m)')
axes[1, 1].set_title('Robot Height')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_{test_axis}_tracking.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to: rl_{test_axis}_tracking.png")
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
