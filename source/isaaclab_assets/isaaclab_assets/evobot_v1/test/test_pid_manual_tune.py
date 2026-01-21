#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manual PID tuning with response plotting for Evobot V1 velocity control.

This script allows manual tuning of PID gains and visualizes the response.
It generates step commands and plots tracking performance to help analyze
PID behavior and optimize gains.

Features:
- Manual PID gain tuning via command line
- Step response testing with configurable commands
- Real-time tracking metrics (rise time, settling time, overshoot, SSE)
- CSV logging for offline analysis
- Auto-generated plotting script
- Support for both linear and angular velocity testing

Usage:
    # Test linear velocity tracking with default PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_manual_tune.py \
        --num_envs 1 --test_type linear

    # Test with custom PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_manual_tune.py \
        --num_envs 1 \
        --kp_linear 2.0 --ki_linear 0.1 --kd_linear 0.5 \
        --test_type linear

    # Test angular velocity with custom step sequence
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_manual_tune.py \
        --num_envs 1 \
        --kp_angular 3.0 --ki_angular 0.2 --kd_angular 0.8 \
        --test_type angular \
        --step_values 0.0,1.0,0.0,-1.0,0.0

    # Test both linear and angular simultaneously
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_manual_tune.py \
        --num_envs 1 \
        --test_type both \
        --step_duration 8.0

Output:
    - Console: Real-time tracking performance with metrics
    - CSV file: logs/pid_manual_tune_<timestamp>.csv
    - Auto-generated Python plotting script

Metrics Computed:
    - Rise time: Time to go from 10% to 90% of command
    - Settling time: Time to settle within ±5% of command
    - Overshoot: Peak overshoot percentage
    - Steady-state error: Average error in final 20% of step

Tuning Tips:
    - Kp: Affects speed of response (higher = faster, but may overshoot)
    - Ki: Eliminates steady-state error (higher = faster convergence, but may oscillate)
    - Kd: Reduces overshoot and oscillations (higher = more damped)
    - Start with low gains and gradually increase Kp until acceptable response
    - Add Ki if steady-state error persists
    - Add Kd if overshoot or oscillations occur
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
parser = argparse.ArgumentParser(description="Manual PID tuning with response analysis for velocity control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recommended: 1 for cleaner plots)")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Velocity", help="Task name")

# PID gains for linear velocity (forward/backward)
parser.add_argument("--kp_linear", type=float, default=1.0, help="Proportional gain for linear velocity")
parser.add_argument("--ki_linear", type=float, default=0.05, help="Integral gain for linear velocity")
parser.add_argument("--kd_linear", type=float, default=0.2, help="Derivative gain for linear velocity")

# PID gains for angular velocity (turning)
parser.add_argument("--kp_angular", type=float, default=2.0, help="Proportional gain for angular velocity")
parser.add_argument("--ki_angular", type=float, default=0.1, help="Integral gain for angular velocity")
parser.add_argument("--kd_angular", type=float, default=0.5, help="Derivative gain for angular velocity")

# Test configuration
parser.add_argument("--test_type", type=str, default="linear",
                    choices=["linear", "angular", "both"],
                    help="Type of velocity to test: linear (vx), angular (wz), or both")
parser.add_argument("--step_duration", type=float, default=6.0,
                    help="Duration of each step command (seconds)")
parser.add_argument("--step_values", type=str, default="0.0,0.5,0.0,-0.5,0.0,0.8,0.0",
                    help="Comma-separated velocity step values (m/s for linear, rad/s for angular)")

# Control parameters
parser.add_argument("--wheel_base", type=float, default=0.2, help="Distance between wheels (m)")
parser.add_argument("--wheel_radius", type=float, default=0.05, help="Wheel radius (m)")
parser.add_argument("--effort_scale", type=float, default=100.0, help="Scale factor for wheel effort")
parser.add_argument("--pid_decimation", type=int, default=4, help="PID update decimation (1 = every step)")

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


class VelocityPIDController:
    """PID controller for velocity tracking with separate gains for linear and angular."""

    def __init__(
        self,
        kp_linear: float,
        ki_linear: float,
        kd_linear: float,
        kp_angular: float,
        ki_angular: float,
        kd_angular: float,
        wheel_base: float,
        wheel_radius: float,
        effort_scale: float,
        num_envs: int,
        device: str = "cuda",
    ):
        """Initialize PID controller.

        Args:
            kp_linear: Proportional gain for linear velocity
            ki_linear: Integral gain for linear velocity
            kd_linear: Derivative gain for linear velocity
            kp_angular: Proportional gain for angular velocity
            ki_angular: Integral gain for angular velocity
            kd_angular: Derivative gain for angular velocity
            wheel_base: Distance between wheels (m)
            wheel_radius: Wheel radius (m)
            effort_scale: Scale factor for wheel efforts
            num_envs: Number of parallel environments
            device: Torch device (cuda/cpu)
        """
        self.kp_linear = kp_linear
        self.ki_linear = ki_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.ki_angular = ki_angular
        self.kd_angular = kd_angular
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.effort_scale = effort_scale
        self.num_envs = num_envs
        self.device = device

        # PID state for linear velocity
        self.error_integral_linear = torch.zeros(num_envs, device=device)
        self.error_prev_linear = torch.zeros(num_envs, device=device)

        # PID state for angular velocity
        self.error_integral_angular = torch.zeros(num_envs, device=device)
        self.error_prev_angular = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset PID state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.error_integral_linear[env_ids] = 0.0
        self.error_prev_linear[env_ids] = 0.0
        self.error_integral_angular[env_ids] = 0.0
        self.error_prev_angular[env_ids] = 0.0

    def compute(self, vel_cmd: torch.Tensor, vel_current: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute wheel efforts from velocity commands using PID control.

        Args:
            vel_cmd: Commanded velocities (num_envs, 2) [vx, wz]
            vel_current: Current velocities (num_envs, 2) [vx, wz]
            dt: Time step (seconds)

        Returns:
            wheel_efforts: Wheel efforts (num_envs, 2) [left, right]
        """
        # Extract velocities
        vx_cmd = vel_cmd[:, 0]
        wz_cmd = vel_cmd[:, 1]
        vx_current = vel_current[:, 0]
        wz_current = vel_current[:, 1]

        # Compute errors
        error_linear = vx_cmd - vx_current
        error_angular = wz_cmd - wz_current

        # Linear velocity PID
        self.error_integral_linear += error_linear * dt
        self.error_integral_linear = torch.clamp(self.error_integral_linear, -10.0, 10.0)  # Anti-windup
        error_derivative_linear = (error_linear - self.error_prev_linear) / dt
        u_linear = (
            self.kp_linear * error_linear
            + self.ki_linear * self.error_integral_linear
            + self.kd_linear * error_derivative_linear
        )
        self.error_prev_linear = error_linear.clone()

        # Angular velocity PID
        self.error_integral_angular += error_angular * dt
        self.error_integral_angular = torch.clamp(self.error_integral_angular, -10.0, 10.0)  # Anti-windup
        error_derivative_angular = (error_angular - self.error_prev_angular) / dt
        u_angular = (
            self.kp_angular * error_angular
            + self.ki_angular * self.error_integral_angular
            + self.kd_angular * error_derivative_angular
        )
        self.error_prev_angular = error_angular.clone()

        # Differential drive kinematics: convert (v, w) to (v_left, v_right)
        # v_left = v - (L/2)*w
        # v_right = v + (L/2)*w
        v_left = u_linear - (self.wheel_base / 2.0) * u_angular
        v_right = u_linear + (self.wheel_base / 2.0) * u_angular

        # Convert to wheel efforts
        effort_left = v_left * self.effort_scale
        effort_right = v_right * self.effort_scale

        # Clamp efforts to reasonable range
        effort_left = torch.clamp(effort_left, -400.0, 400.0)
        effort_right = torch.clamp(effort_right, -400.0, 400.0)

        # Stack into (num_envs, 2)
        wheel_efforts = torch.stack([effort_left, effort_right], dim=-1)

        return wheel_efforts


class ResponseLogger:
    """Logger for velocity tracking data with CSV export."""

    def __init__(self, log_dir: str = "logs", filename_prefix: str = "pid_manual_tune"):
        """Initialize logger and create CSV file."""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")

        # Create CSV file with header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "vx_cmd", "vx_actual", "vx_error",
                "wz_cmd", "wz_actual", "wz_error",
                "effort_left", "effort_right"
            ])

        print(f"[INFO] Logging to: {self.filepath}")

    def log(
        self,
        time: float,
        vx_cmd: float,
        vx_actual: float,
        wz_cmd: float,
        wz_actual: float,
        effort_left: float,
        effort_right: float,
    ):
        """Log single data point to CSV."""
        vx_error = vx_cmd - vx_actual
        wz_error = wz_cmd - wz_actual

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{time:.4f}",
                f"{vx_cmd:.6f}", f"{vx_actual:.6f}", f"{vx_error:.6f}",
                f"{wz_cmd:.6f}", f"{wz_actual:.6f}", f"{wz_error:.6f}",
                f"{effort_left:.2f}", f"{effort_right:.2f}",
            ])

    def get_filepath(self) -> str:
        """Get path to CSV file."""
        return self.filepath


def compute_step_response_metrics(time_data, cmd_data, actual_data, step_times):
    """Compute standard step response metrics for each step.

    Metrics:
        - Rise time (10% to 90%)
        - Settling time (within 5% of final value)
        - Overshoot (%)
        - Steady-state error (average of last 20%)

    Args:
        time_data: Array of time values
        cmd_data: Array of command values
        actual_data: Array of actual values
        step_times: List of step transition times

    Returns:
        List of metric dictionaries
    """
    metrics = []

    for i in range(len(step_times) - 1):
        t_start = step_times[i]
        t_end = step_times[i + 1]

        # Get data for this step
        mask = (time_data >= t_start) & (time_data < t_end)
        t_step = time_data[mask] - t_start
        cmd_step = cmd_data[mask]
        actual_step = actual_data[mask]

        if len(t_step) < 20:
            continue

        # Get command value (should be constant during step)
        cmd_value = cmd_step[0]

        # Skip zero commands or very small commands
        if abs(cmd_value) < 0.01:
            continue

        # Rise time (10% to 90% of command value)
        threshold_10 = abs(cmd_value) * 0.1
        threshold_90 = abs(cmd_value) * 0.9

        idx_10 = np.where(np.abs(actual_step) >= threshold_10)[0]
        idx_90 = np.where(np.abs(actual_step) >= threshold_90)[0]

        rise_time = None
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = t_step[idx_90[0]] - t_step[idx_10[0]]

        # Settling time (within 5% of final value and stays there)
        final_value = cmd_value
        threshold_settle = abs(final_value) * 0.05
        settled_mask = np.abs(actual_step - final_value) <= threshold_settle

        settling_time = None
        if np.any(settled_mask):
            # Find first time it settles and stays settled for 10+ samples
            for j in range(len(t_step) - 10):
                if np.all(settled_mask[j : j + 10]):
                    settling_time = t_step[j]
                    break

        # Overshoot (%)
        overshoot_pct = 0.0
        if cmd_value != 0:
            if cmd_value > 0:
                max_value = np.max(actual_step)
                if max_value > cmd_value:
                    overshoot_pct = (max_value - cmd_value) / cmd_value * 100.0
            else:
                min_value = np.min(actual_step)
                if min_value < cmd_value:
                    overshoot_pct = (cmd_value - min_value) / abs(cmd_value) * 100.0

        # Steady-state error (average of last 20% of data)
        steady_idx = int(len(actual_step) * 0.8)
        steady_state_error = np.mean(cmd_step[steady_idx:] - actual_step[steady_idx:])

        metrics.append({
            "step": i,
            "command": cmd_value,
            "rise_time": rise_time,
            "settling_time": settling_time,
            "overshoot": overshoot_pct,
            "steady_state_error": steady_state_error,
        })

    return metrics


def print_metrics(metrics, title: str):
    """Pretty print step response metrics."""
    if not metrics:
        return

    print(f"\n{title}")
    print("-" * 80)
    for m in metrics:
        print(f"  Step {m['step']}: Command = {m['command']:.3f}")
        if m["rise_time"] is not None:
            print(f"    ├─ Rise time:         {m['rise_time']:.3f} s")
        if m["settling_time"] is not None:
            print(f"    ├─ Settling time:     {m['settling_time']:.3f} s")
        print(f"    ├─ Overshoot:         {m['overshoot']:.2f} %")
        print(f"    └─ Steady-state err:  {m['steady_state_error']:.4f}")
        print()


def main():
    """Main function."""

    # Parse step values
    step_values = [float(x) for x in args_cli.step_values.split(",")]

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # Calculate total test duration
    total_duration = len(step_values) * args_cli.step_duration
    env_cfg.episode_length_s = total_duration + 10.0  # Add buffer

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print configuration
    print("\n" + "=" * 80)
    print("EVOBOT V1 - MANUAL PID TUNING WITH RESPONSE ANALYSIS")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Test type: {args_cli.test_type.upper()}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Test duration: {total_duration:.1f} seconds")
    print(f"Step duration: {args_cli.step_duration:.1f} seconds")
    print(f"Step values: {step_values}")
    print("\nPID GAINS:")
    print(f"  Linear  (vx): Kp={args_cli.kp_linear:.4f}, Ki={args_cli.ki_linear:.4f}, Kd={args_cli.kd_linear:.4f}")
    print(f"  Angular (wz): Kp={args_cli.kp_angular:.4f}, Ki={args_cli.ki_angular:.4f}, Kd={args_cli.kd_angular:.4f}")
    print("\nCONTROL PARAMETERS:")
    print(f"  Wheel base: {args_cli.wheel_base} m")
    print(f"  Wheel radius: {args_cli.wheel_radius} m")
    print(f"  Effort scale: {args_cli.effort_scale}")
    print(f"  PID decimation: {args_cli.pid_decimation}")
    print("=" * 80)

    # Create PID controller
    pid_controller = VelocityPIDController(
        kp_linear=args_cli.kp_linear,
        ki_linear=args_cli.ki_linear,
        kd_linear=args_cli.kd_linear,
        kp_angular=args_cli.kp_angular,
        ki_angular=args_cli.ki_angular,
        kd_angular=args_cli.kd_angular,
        wheel_base=args_cli.wheel_base,
        wheel_radius=args_cli.wheel_radius,
        effort_scale=args_cli.effort_scale,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    # Initialize logger
    logger = ResponseLogger()

    # Reset environment and controller
    env.reset()
    pid_controller.reset()

    # Get robot asset
    robot = env.unwrapped.scene["robot"]

    # Get timestep
    dt_sim = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation
    dt_pid = dt_sim * args_cli.pid_decimation

    # Get wheel joint indices
    wheel_indices = [
        robot.joint_names.index("left_wheel_joint"),
        robot.joint_names.index("right_wheel_joint")
    ]

    # Data storage
    time_log = []
    vx_cmd_log = []
    vx_actual_log = []
    wz_cmd_log = []
    wz_actual_log = []
    effort_left_log = []
    effort_right_log = []
    step_times = [0.0]

    # Main loop variables
    step_count = 0
    pid_step_counter = 0
    elapsed_time = 0.0
    current_step_idx = 0

    # Initial command
    vel_cmd = torch.zeros(args_cli.num_envs, 2, device=args_cli.device)
    if args_cli.test_type in ["linear", "both"]:
        vel_cmd[:, 0] = step_values[current_step_idx]
    if args_cli.test_type in ["angular", "both"]:
        vel_cmd[:, 1] = step_values[current_step_idx]

    wheel_efforts = torch.zeros(args_cli.num_envs, 2, device=args_cli.device)

    print("\n[INFO] Starting velocity tracking test...")
    print(f"\n{'Time':>8s} {'VxCmd':>10s} {'VxAct':>10s} {'WzCmd':>10s} {'WzAct':>10s} {'VxErr':>10s} {'WzErr':>10s}")
    print("-" * 78)

    try:
        while simulation_app.is_running() and elapsed_time < total_duration:
            with torch.inference_mode():
                # Update command based on time
                if elapsed_time >= (current_step_idx + 1) * args_cli.step_duration:
                    current_step_idx += 1
                    if current_step_idx < len(step_values):
                        if args_cli.test_type in ["linear", "both"]:
                            vel_cmd[:, 0] = step_values[current_step_idx]
                        if args_cli.test_type in ["angular", "both"]:
                            vel_cmd[:, 1] = step_values[current_step_idx]
                        step_times.append(elapsed_time)
                        print(f"\n[STEP {current_step_idx}] New command: vx={vel_cmd[0,0]:.2f}, wz={vel_cmd[0,1]:.2f}\n")

                # Get current velocity from robot
                vel_current = torch.stack([
                    robot.data.root_lin_vel_b[:, 0],  # vx in base frame
                    robot.data.root_ang_vel_b[:, 2],  # wz in base frame
                ], dim=-1)

                # Update PID at decimated rate
                if pid_step_counter % args_cli.pid_decimation == 0:
                    wheel_efforts = pid_controller.compute(vel_cmd, vel_current, dt_pid)

                # Apply wheel efforts directly to robot
                joint_efforts = torch.zeros(args_cli.num_envs, robot.num_joints, device=args_cli.device)
                joint_efforts[:, wheel_indices[0]] = wheel_efforts[:, 0]
                joint_efforts[:, wheel_indices[1]] = wheel_efforts[:, 1]
                robot.set_joint_effort_target(joint_efforts)
                robot.write_data_to_sim()

            # Step environment (zero action since efforts already applied)
            obs, reward, terminated, truncated, info = env.step(
                torch.zeros(args_cli.num_envs, env.action_space.shape[0], device=args_cli.device)
            )

            with torch.inference_mode():
                # Log data
                if step_count % args_cli.log_interval == 0:
                    vx_cmd_val = vel_cmd[0, 0].item()
                    wz_cmd_val = vel_cmd[0, 1].item()
                    vx_actual_val = vel_current[0, 0].item()
                    wz_actual_val = vel_current[0, 1].item()
                    effort_left_val = wheel_efforts[0, 0].item()
                    effort_right_val = wheel_efforts[0, 1].item()

                    time_log.append(elapsed_time)
                    vx_cmd_log.append(vx_cmd_val)
                    vx_actual_log.append(vx_actual_val)
                    wz_cmd_log.append(wz_cmd_val)
                    wz_actual_log.append(wz_actual_val)
                    effort_left_log.append(effort_left_val)
                    effort_right_log.append(effort_right_val)

                    logger.log(
                        elapsed_time,
                        vx_cmd_val, vx_actual_val,
                        wz_cmd_val, wz_actual_val,
                        effort_left_val, effort_right_val
                    )

                    # Print progress
                    if step_count % args_cli.print_interval == 0:
                        vx_err = vx_cmd_val - vx_actual_val
                        wz_err = wz_cmd_val - wz_actual_val
                        print(
                            f"{elapsed_time:8.2f} {vx_cmd_val:10.3f} {vx_actual_val:10.3f} "
                            f"{wz_cmd_val:10.3f} {wz_actual_val:10.3f} {vx_err:10.3f} {wz_err:10.3f}"
                        )

            step_count += 1
            pid_step_counter += 1
            elapsed_time += dt_sim

    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")

    # Compute metrics
    print("\n" + "=" * 80)
    print("STEP RESPONSE METRICS")
    print("=" * 80)

    time_array = np.array(time_log)
    vx_cmd_array = np.array(vx_cmd_log)
    vx_actual_array = np.array(vx_actual_log)
    wz_cmd_array = np.array(wz_cmd_log)
    wz_actual_array = np.array(wz_actual_log)

    if args_cli.test_type in ["linear", "both"]:
        vx_metrics = compute_step_response_metrics(time_array, vx_cmd_array, vx_actual_array, step_times)
        print_metrics(vx_metrics, "LINEAR VELOCITY (vx) TRACKING:")

    if args_cli.test_type in ["angular", "both"]:
        wz_metrics = compute_step_response_metrics(time_array, wz_cmd_array, wz_actual_array, step_times)
        print_metrics(wz_metrics, "ANGULAR VELOCITY (wz) TRACKING:")

    print("=" * 80)

    # Generate plotting script
    print("\n" + "=" * 80)
    print("PLOT RESPONSE CURVES WITH THIS PYTHON SCRIPT:")
    print("=" * 80)
    print(f"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('{logger.get_filepath()}')

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Linear velocity tracking
axes[0].plot(df['time'], df['vx_cmd'], 'r--', linewidth=2, label='Command', alpha=0.8)
axes[0].plot(df['time'], df['vx_actual'], 'b-', linewidth=1.5, label='Actual')
axes[0].fill_between(df['time'], df['vx_cmd'] * 0.95, df['vx_cmd'] * 1.05,
                      color='red', alpha=0.1, label='±5% band')
axes[0].set_ylabel('Linear Velocity (m/s)', fontsize=12)
axes[0].set_title(f'Linear Velocity Tracking (Kp={args_cli.kp_linear}, Ki={args_cli.ki_linear}, Kd={args_cli.kd_linear})',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Angular velocity tracking
axes[1].plot(df['time'], df['wz_cmd'], 'r--', linewidth=2, label='Command', alpha=0.8)
axes[1].plot(df['time'], df['wz_actual'], 'b-', linewidth=1.5, label='Actual')
axes[1].fill_between(df['time'], df['wz_cmd'] * 0.95, df['wz_cmd'] * 1.05,
                      color='red', alpha=0.1, label='±5% band')
axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
axes[1].set_title(f'Angular Velocity Tracking (Kp={args_cli.kp_angular}, Ki={args_cli.ki_angular}, Kd={args_cli.kd_angular})',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10, loc='upper right')
axes[1].grid(True, alpha=0.3)

# Control efforts
axes[2].plot(df['time'], df['effort_left'], 'g-', linewidth=1.2, label='Left wheel', alpha=0.8)
axes[2].plot(df['time'], df['effort_right'], 'm-', linewidth=1.2, label='Right wheel', alpha=0.8)
axes[2].axhline(y=400, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Effort limits')
axes[2].axhline(y=-400, color='r', linestyle='--', linewidth=1, alpha=0.5)
axes[2].set_xlabel('Time (s)', fontsize=12)
axes[2].set_ylabel('Wheel Effort', fontsize=12)
axes[2].set_title('Control Efforts', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10, loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pid_response_analysis.png', dpi=300, bbox_inches='tight')
print("\\nPlot saved to: pid_response_analysis.png")
plt.show()
""")
    print("=" * 80)

    # Print command to re-run with different gains
    print("\n" + "=" * 80)
    print("ADJUST PID GAINS AND RE-RUN:")
    print("=" * 80)
    print("./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_manual_tune.py \\")
    print(f"    --num_envs {args_cli.num_envs} \\")
    print(f"    --test_type {args_cli.test_type} \\")
    print(f"    --kp_linear <new_kp> --ki_linear <new_ki> --kd_linear <new_kd> \\")
    print(f"    --kp_angular <new_kp> --ki_angular <new_ki> --kd_angular <new_kd>")
    print("=" * 80)

    # Close environment
    print("\n[INFO] Closing environment...")
    env.close()
    print(f"[INFO] Data saved to: {logger.get_filepath()}")
    print("[INFO] Test completed successfully!")


if __name__ == "__main__":
    main()
    simulation_app.close()
