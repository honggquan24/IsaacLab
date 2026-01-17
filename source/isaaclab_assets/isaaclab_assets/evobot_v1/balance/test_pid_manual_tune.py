#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manual PID tuning with response plotting for Evobot V1 balance control.

This script allows manual tuning of PID gains for balance (roll/pitch control)
and visualizes the step response to help analyze and optimize controller performance.

Features:
- Manual PID gain tuning for roll and pitch control
- Step response testing with configurable disturbances
- Real-time tracking metrics (rise time, settling time, overshoot, SSE)
- CSV logging for offline analysis
- Auto-generated plotting script with multiple subplots
- Separate gains for roll and pitch axes

Balance Control:
- Roll: Side-to-side tilt (differential wheel control)
- Pitch: Forward-backward tilt (common wheel control)
- Target: Keep robot upright (roll=0, pitch=0)

Usage:
    # Test with default PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_manual_tune.py \
        --num_envs 1

    # Test with custom PID gains for roll and pitch
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_manual_tune.py \
        --num_envs 1 \
        --kp_roll 1.5 --ki_roll 0.05 --kd_roll 0.3 \
        --kp_pitch 2.0 --ki_pitch 0.1 --kd_pitch 0.5

    # Test with external disturbance
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_manual_tune.py \
        --num_envs 1 \
        --apply_disturbance \
        --disturbance_magnitude 0.2

    # Longer test duration for settling analysis
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_manual_tune.py \
        --num_envs 1 \
        --test_duration 20.0

Output:
    - Console: Real-time orientation and control effort
    - CSV file: logs/balance_pid_manual_tune_<timestamp>.csv
    - Auto-generated Python plotting script

Metrics Computed:
    - Rise time: Time to go from 10% to 90% of target
    - Settling time: Time to settle within ±2° of upright
    - Overshoot: Maximum deviation from target (degrees)
    - Steady-state error: Average error in final 20% of test

Tuning Tips:
    - Kp: Increases stiffness (higher = faster response, but may oscillate)
    - Ki: Eliminates steady-state error (add slowly to avoid instability)
    - Kd: Adds damping (higher = less overshoot, smoother response)
    - Start with Kp only, then add Kd to reduce oscillations
    - Add Ki last if steady-state error persists
    - Tune roll and pitch separately (they affect different wheel combinations)
"""

import argparse
import torch
import os
import csv
import numpy as np
from datetime import datetime
from typing import Optional

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Manual PID tuning with response analysis for balance control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recommended: 1 for analysis)")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Balance", help="Task name")

# PID gains for roll (side-to-side balance)
parser.add_argument("--kp_roll", type=float, default=1.0, help="Proportional gain for roll control")
parser.add_argument("--ki_roll", type=float, default=0.02, help="Integral gain for roll control")
parser.add_argument("--kd_roll", type=float, default=0.3, help="Derivative gain for roll control")

# PID gains for pitch (forward-backward balance)
parser.add_argument("--kp_pitch", type=float, default=1.5, help="Proportional gain for pitch control")
parser.add_argument("--ki_pitch", type=float, default=0.05, help="Integral gain for pitch control")
parser.add_argument("--kd_pitch", type=float, default=0.4, help="Derivative gain for pitch control")

# Test configuration
parser.add_argument("--test_duration", type=float, default=15.0, help="Total test duration (seconds)")
parser.add_argument("--apply_disturbance", action="store_true", help="Apply external disturbance during test")
parser.add_argument("--disturbance_time", type=float, default=5.0, help="Time to apply disturbance (seconds)")
parser.add_argument("--disturbance_magnitude", type=float, default=0.15, help="Disturbance magnitude (radians)")
parser.add_argument("--disturbance_duration", type=float, default=0.5, help="Disturbance duration (seconds)")

# Control parameters
parser.add_argument("--effort_scale", type=float, default=1.0, help="Scale factor for control effort")
parser.add_argument("--target_roll", type=float, default=0.0, help="Target roll angle (radians)")
parser.add_argument("--target_pitch", type=float, default=0.0, help="Target pitch angle (radians)")

# Logging
parser.add_argument("--log_interval", type=int, default=1, help="Log data every N steps")
parser.add_argument("--print_interval", type=int, default=50, help="Print status every N steps")

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
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat


class BalancePIDController:
    """PID controller for balance control (roll and pitch)."""

    def __init__(
        self,
        kp_roll: float,
        ki_roll: float,
        kd_roll: float,
        kp_pitch: float,
        ki_pitch: float,
        kd_pitch: float,
        effort_scale: float,
        num_envs: int,
        device: str = "cuda",
    ):
        """Initialize balance PID controller.

        Args:
            kp_roll: Proportional gain for roll
            ki_roll: Integral gain for roll
            kd_roll: Derivative gain for roll
            kp_pitch: Proportional gain for pitch
            ki_pitch: Integral gain for pitch
            kd_pitch: Derivative gain for pitch
            effort_scale: Scale factor for control efforts
            num_envs: Number of parallel environments
            device: Torch device (cuda/cpu)
        """
        self.kp_roll = kp_roll
        self.ki_roll = ki_roll
        self.kd_roll = kd_roll
        self.kp_pitch = kp_pitch
        self.ki_pitch = ki_pitch
        self.kd_pitch = kd_pitch
        self.effort_scale = effort_scale
        self.num_envs = num_envs
        self.device = device

        # PID state for roll
        self.roll_integral = torch.zeros(num_envs, device=device)
        self.roll_prev_error = torch.zeros(num_envs, device=device)

        # PID state for pitch
        self.pitch_integral = torch.zeros(num_envs, device=device)
        self.pitch_prev_error = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset PID state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.roll_integral[env_ids] = 0.0
        self.roll_prev_error[env_ids] = 0.0
        self.pitch_integral[env_ids] = 0.0
        self.pitch_prev_error[env_ids] = 0.0

    def compute(
        self,
        current_rpy: torch.Tensor,
        target_rpy: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute control actions for balance.

        Args:
            current_rpy: Current orientation (num_envs, 3) [roll, pitch, yaw]
            target_rpy: Target orientation (num_envs, 3) [roll, pitch, yaw]
            dt: Time step (seconds)

        Returns:
            actions: Control actions (num_envs, 5) [left_wheel, right_wheel, arm, gripper_l, gripper_r]
        """
        # Extract roll and pitch
        roll_current = current_rpy[:, 0]
        pitch_current = current_rpy[:, 1]
        roll_target = target_rpy[:, 0]
        pitch_target = target_rpy[:, 1]

        # Compute errors
        roll_error = roll_target - roll_current
        pitch_error = pitch_target - pitch_current

        # Roll PID (controls differential wheel speed)
        self.roll_integral += roll_error * dt
        self.roll_integral = torch.clamp(self.roll_integral, -10.0, 10.0)  # Anti-windup
        roll_derivative = (roll_error - self.roll_prev_error) / dt
        roll_output = (
            self.kp_roll * roll_error
            + self.ki_roll * self.roll_integral
            + self.kd_roll * roll_derivative
        )
        self.roll_prev_error = roll_error.clone()

        # Pitch PID (controls common wheel speed)
        self.pitch_integral += pitch_error * dt
        self.pitch_integral = torch.clamp(self.pitch_integral, -10.0, 10.0)  # Anti-windup
        pitch_derivative = (pitch_error - self.pitch_prev_error) / dt
        pitch_output = (
            self.kp_pitch * pitch_error
            + self.ki_pitch * self.pitch_integral
            + self.kd_pitch * pitch_derivative
        )
        self.pitch_prev_error = pitch_error.clone()

        # Convert to wheel commands
        # Pitch controls both wheels equally (forward/backward)
        # Roll controls wheels differentially (left/right balance)
        wheel_left = (pitch_output + roll_output) * self.effort_scale
        wheel_right = (pitch_output - roll_output) * self.effort_scale

        # Zero effort for arm and grippers
        arm_effort = torch.zeros(self.num_envs, device=self.device)
        gripper_left = torch.zeros(self.num_envs, device=self.device)
        gripper_right = torch.zeros(self.num_envs, device=self.device)

        # Stack into action tensor
        actions = torch.stack([wheel_left, wheel_right, arm_effort, gripper_left, gripper_right], dim=-1)
        actions = torch.clamp(actions, -1.0, 1.0)

        return actions


class BalanceResponseLogger:
    """Logger for balance control data with CSV export."""

    def __init__(self, log_dir: str = "logs", filename_prefix: str = "balance_pid_manual_tune"):
        """Initialize logger and create CSV file."""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")

        # Create CSV file with header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "roll_target", "roll_actual", "roll_error",
                "pitch_target", "pitch_actual", "pitch_error",
                "action_left", "action_right"
            ])

        print(f"[INFO] Logging to: {self.filepath}")

    def log(
        self,
        time: float,
        roll_target: float,
        roll_actual: float,
        pitch_target: float,
        pitch_actual: float,
        action_left: float,
        action_right: float,
    ):
        """Log single data point to CSV."""
        roll_error = roll_target - roll_actual
        pitch_error = pitch_target - pitch_actual

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{time:.4f}",
                f"{roll_target:.6f}", f"{roll_actual:.6f}", f"{roll_error:.6f}",
                f"{pitch_target:.6f}", f"{pitch_actual:.6f}", f"{pitch_error:.6f}",
                f"{action_left:.6f}", f"{action_right:.6f}",
            ])

    def get_filepath(self) -> str:
        """Get path to CSV file."""
        return self.filepath


def compute_balance_metrics(time_data, target_data, actual_data, disturbance_time=None):
    """Compute balance performance metrics.

    Metrics:
        - Settling time (within ±2° of target)
        - Maximum deviation (degrees)
        - Steady-state error (average of last 20%)
        - Recovery time (if disturbance applied)

    Args:
        time_data: Array of time values
        target_data: Array of target angles (radians)
        actual_data: Array of actual angles (radians)
        disturbance_time: Time when disturbance was applied (optional)

    Returns:
        Dictionary of metrics
    """
    # Convert to degrees for easier interpretation
    target_deg = np.rad2deg(target_data)
    actual_deg = np.rad2deg(actual_data)
    error_deg = target_deg - actual_deg

    # Settling time (within ±2° of target)
    settling_threshold = 2.0  # degrees
    settled_mask = np.abs(error_deg) <= settling_threshold

    settling_time = None
    if np.any(settled_mask):
        # Find first time it settles and stays settled for 1 second (60 samples at 60Hz)
        samples_to_check = min(60, len(time_data) // 10)
        for i in range(len(time_data) - samples_to_check):
            if np.all(settled_mask[i : i + samples_to_check]):
                settling_time = time_data[i]
                break

    # Maximum deviation
    max_deviation = np.max(np.abs(error_deg))

    # Steady-state error (last 20% of data)
    steady_idx = int(len(error_deg) * 0.8)
    steady_state_error = np.mean(error_deg[steady_idx:])

    # Recovery time (if disturbance applied)
    recovery_time = None
    if disturbance_time is not None:
        # Find data after disturbance
        post_disturbance_mask = time_data > disturbance_time + 0.5  # After disturbance ends
        if np.any(post_disturbance_mask):
            post_disturbance_idx = np.where(post_disturbance_mask)[0]
            post_disturbance_settled = settled_mask[post_disturbance_idx]

            if np.any(post_disturbance_settled):
                for i in range(len(post_disturbance_idx) - samples_to_check):
                    idx_start = post_disturbance_idx[i]
                    if np.all(settled_mask[idx_start : idx_start + samples_to_check]):
                        recovery_time = time_data[idx_start] - disturbance_time
                        break

    return {
        "settling_time": settling_time,
        "max_deviation": max_deviation,
        "steady_state_error": steady_state_error,
        "recovery_time": recovery_time,
    }


def print_balance_metrics(metrics, title: str):
    """Pretty print balance metrics."""
    print(f"\n{title}")
    print("-" * 80)
    if metrics["settling_time"] is not None:
        print(f"  ├─ Settling time:        {metrics['settling_time']:.3f} s")
    else:
        print(f"  ├─ Settling time:        Did not settle within test duration")
    print(f"  ├─ Max deviation:        {metrics['max_deviation']:.2f}°")
    print(f"  ├─ Steady-state error:   {metrics['steady_state_error']:.3f}°")
    if metrics["recovery_time"] is not None:
        print(f"  └─ Recovery time:        {metrics['recovery_time']:.3f} s (after disturbance)")
    print()


def main():
    """Main function."""

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # Set episode length
    env_cfg.episode_length_s = args_cli.test_duration + 5.0

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print configuration
    print("\n" + "=" * 80)
    print("EVOBOT V1 - MANUAL BALANCE PID TUNING")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Test duration: {args_cli.test_duration:.1f} seconds")
    print(f"Target orientation: roll={np.rad2deg(args_cli.target_roll):.1f}°, pitch={np.rad2deg(args_cli.target_pitch):.1f}°")
    if args_cli.apply_disturbance:
        print(f"Disturbance: {np.rad2deg(args_cli.disturbance_magnitude):.1f}° at t={args_cli.disturbance_time:.1f}s for {args_cli.disturbance_duration:.1f}s")
    print("\nPID GAINS:")
    print(f"  Roll:  Kp={args_cli.kp_roll:.4f}, Ki={args_cli.ki_roll:.4f}, Kd={args_cli.kd_roll:.4f}")
    print(f"  Pitch: Kp={args_cli.kp_pitch:.4f}, Ki={args_cli.ki_pitch:.4f}, Kd={args_cli.kd_pitch:.4f}")
    print("=" * 80)

    # Create PID controller
    pid_controller = BalancePIDController(
        kp_roll=args_cli.kp_roll,
        ki_roll=args_cli.ki_roll,
        kd_roll=args_cli.kd_roll,
        kp_pitch=args_cli.kp_pitch,
        ki_pitch=args_cli.ki_pitch,
        kd_pitch=args_cli.kd_pitch,
        effort_scale=args_cli.effort_scale,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    # Initialize logger
    logger = BalanceResponseLogger()

    # Reset environment and controller
    env.reset()
    pid_controller.reset()

    # Get robot asset
    robot = env.unwrapped.scene["robot"]

    # Get timestep
    dt = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation

    # Target orientation (upright)
    target_rpy = torch.zeros(args_cli.num_envs, 3, device=args_cli.device)
    target_rpy[:, 0] = args_cli.target_roll
    target_rpy[:, 1] = args_cli.target_pitch

    # Data storage
    time_log = []
    roll_target_log = []
    roll_actual_log = []
    pitch_target_log = []
    pitch_actual_log = []
    action_left_log = []
    action_right_log = []

    # Main loop variables
    step_count = 0
    elapsed_time = 0.0
    disturbance_applied = False

    print("\n[INFO] Starting balance control test...")
    print(f"\n{'Time':>8s} {'RollTgt':>10s} {'RollAct':>10s} {'PitchTgt':>10s} {'PitchAct':>10s} {'RollErr':>10s} {'PitchErr':>10s}")
    print("-" * 78)

    try:
        while simulation_app.is_running() and elapsed_time < args_cli.test_duration:
            with torch.inference_mode():
                # Get current orientation
                quat = robot.data.root_quat_w
                roll, pitch, yaw = euler_xyz_from_quat(quat)
                current_rpy = torch.stack([roll, pitch, yaw], dim=-1)

                # Apply disturbance if specified
                if (args_cli.apply_disturbance and
                    not disturbance_applied and
                    elapsed_time >= args_cli.disturbance_time and
                    elapsed_time < args_cli.disturbance_time + args_cli.disturbance_duration):

                    # Apply impulse to robot base
                    if not disturbance_applied:
                        print(f"\n[DISTURBANCE] Applying {np.rad2deg(args_cli.disturbance_magnitude):.1f}° impulse\n")
                        disturbance_applied = True

                    # Modify target temporarily to simulate disturbance
                    disturbed_target = target_rpy.clone()
                    disturbed_target[:, 0] += args_cli.disturbance_magnitude  # Roll disturbance
                    actions = pid_controller.compute(current_rpy, disturbed_target, dt)
                else:
                    # Normal PID control
                    actions = pid_controller.compute(current_rpy, target_rpy, dt)

            # Step environment
            _, _, _, _, _ = env.step(actions)

            with torch.inference_mode():
                # Log data
                if step_count % args_cli.log_interval == 0:
                    roll_target_val = target_rpy[0, 0].item()
                    roll_actual_val = current_rpy[0, 0].item()
                    pitch_target_val = target_rpy[0, 1].item()
                    pitch_actual_val = current_rpy[0, 1].item()
                    action_left_val = actions[0, 0].item()
                    action_right_val = actions[0, 1].item()

                    time_log.append(elapsed_time)
                    roll_target_log.append(roll_target_val)
                    roll_actual_log.append(roll_actual_val)
                    pitch_target_log.append(pitch_target_val)
                    pitch_actual_log.append(pitch_actual_val)
                    action_left_log.append(action_left_val)
                    action_right_log.append(action_right_val)

                    logger.log(
                        elapsed_time,
                        roll_target_val, roll_actual_val,
                        pitch_target_val, pitch_actual_val,
                        action_left_val, action_right_val
                    )

                    # Print progress
                    if step_count % args_cli.print_interval == 0:
                        roll_err = np.rad2deg(roll_target_val - roll_actual_val)
                        pitch_err = np.rad2deg(pitch_target_val - pitch_actual_val)
                        print(
                            f"{elapsed_time:8.2f} {np.rad2deg(roll_target_val):10.2f} {np.rad2deg(roll_actual_val):10.2f} "
                            f"{np.rad2deg(pitch_target_val):10.2f} {np.rad2deg(pitch_actual_val):10.2f} "
                            f"{roll_err:10.2f} {pitch_err:10.2f}"
                        )

            step_count += 1
            elapsed_time += dt

    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")

    # Compute metrics
    print("\n" + "=" * 80)
    print("BALANCE PERFORMANCE METRICS")
    print("=" * 80)

    time_array = np.array(time_log)
    roll_target_array = np.array(roll_target_log)
    roll_actual_array = np.array(roll_actual_log)
    pitch_target_array = np.array(pitch_target_log)
    pitch_actual_array = np.array(pitch_actual_log)

    # Compute metrics for roll and pitch
    disturbance_time = args_cli.disturbance_time if args_cli.apply_disturbance else None
    roll_metrics = compute_balance_metrics(time_array, roll_target_array, roll_actual_array, disturbance_time)
    pitch_metrics = compute_balance_metrics(time_array, pitch_target_array, pitch_actual_array, disturbance_time)

    print_balance_metrics(roll_metrics, "ROLL CONTROL:")
    print_balance_metrics(pitch_metrics, "PITCH CONTROL:")

    print("=" * 80)

    # Generate plotting script
    print("\n" + "=" * 80)
    print("PLOT BALANCE RESPONSE WITH THIS PYTHON SCRIPT:")
    print("=" * 80)
    print(f"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('{logger.get_filepath()}')

# Convert radians to degrees
df['roll_target_deg'] = np.rad2deg(df['roll_target'])
df['roll_actual_deg'] = np.rad2deg(df['roll_actual'])
df['roll_error_deg'] = np.rad2deg(df['roll_error'])
df['pitch_target_deg'] = np.rad2deg(df['pitch_target'])
df['pitch_actual_deg'] = np.rad2deg(df['pitch_actual'])
df['pitch_error_deg'] = np.rad2deg(df['pitch_error'])

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Roll response
axes[0].plot(df['time'], df['roll_target_deg'], 'r--', linewidth=2, label='Target', alpha=0.8)
axes[0].plot(df['time'], df['roll_actual_deg'], 'b-', linewidth=1.5, label='Actual')
axes[0].fill_between(df['time'],
                      df['roll_target_deg'] - 2,
                      df['roll_target_deg'] + 2,
                      color='red', alpha=0.1, label='±2° band')
axes[0].axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
axes[0].set_ylabel('Roll Angle (°)', fontsize=12)
axes[0].set_title(f'Roll Control (Kp={args_cli.kp_roll}, Ki={args_cli.ki_roll}, Kd={args_cli.kd_roll})',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Pitch response
axes[1].plot(df['time'], df['pitch_target_deg'], 'r--', linewidth=2, label='Target', alpha=0.8)
axes[1].plot(df['time'], df['pitch_actual_deg'], 'b-', linewidth=1.5, label='Actual')
axes[1].fill_between(df['time'],
                      df['pitch_target_deg'] - 2,
                      df['pitch_target_deg'] + 2,
                      color='red', alpha=0.1, label='±2° band')
axes[1].axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
axes[1].set_ylabel('Pitch Angle (°)', fontsize=12)
axes[1].set_title(f'Pitch Control (Kp={args_cli.kp_pitch}, Ki={args_cli.ki_pitch}, Kd={args_cli.kd_pitch})',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10, loc='upper right')
axes[1].grid(True, alpha=0.3)

# Control actions
axes[2].plot(df['time'], df['action_left'], 'g-', linewidth=1.2, label='Left wheel', alpha=0.8)
axes[2].plot(df['time'], df['action_right'], 'm-', linewidth=1.2, label='Right wheel', alpha=0.8)
axes[2].axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Action limits')
axes[2].axhline(y=-1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
axes[2].axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
axes[2].set_xlabel('Time (s)', fontsize=12)
axes[2].set_ylabel('Control Action', fontsize=12)
axes[2].set_title('Control Actions', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10, loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('balance_pid_response.png', dpi=300, bbox_inches='tight')
print("\\nPlot saved to: balance_pid_response.png")
plt.show()
""")
    print("=" * 80)

    # Print command to re-run with different gains
    print("\n" + "=" * 80)
    print("ADJUST PID GAINS AND RE-RUN:")
    print("=" * 80)
    print("./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_manual_tune.py \\")
    print(f"    --num_envs {args_cli.num_envs} \\")
    print(f"    --kp_roll <new_kp> --ki_roll <new_ki> --kd_roll <new_kd> \\")
    print(f"    --kp_pitch <new_kp> --ki_pitch <new_ki> --kd_pitch <new_kd>")
    print("=" * 80)

    # Close environment
    print("\n[INFO] Closing environment...")
    env.close()
    print(f"[INFO] Data saved to: {logger.get_filepath()}")
    print("[INFO] Test completed successfully!")


if __name__ == "__main__":
    main()
    simulation_app.close()
