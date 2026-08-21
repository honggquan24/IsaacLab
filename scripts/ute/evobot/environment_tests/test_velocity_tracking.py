# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test velocity tracking performance and plot response curves.

This script tests how well the robot tracks velocity commands.
It generates step commands and logs both commanded and actual velocities.
Output CSV can be used to plot tracking response, settling time, overshoot, etc.

Usage:
    # Test with RL policy
    ./isaaclab.sh -p scripts/ute/evobot/environment_tests/test_velocity_tracking.py \
        --load_run 2026-01-15_01-19-39 \
        --checkpoint model_500.pt

    # Test with PID controller
    ./isaaclab.sh -p scripts/ute/evobot/environment_tests/test_velocity_tracking.py \
        --use_pid \
        --kp 1.0 --ki 0.1 --kd 0.5

    # Custom test sequence
    ./isaaclab.sh -p scripts/ute/evobot/environment_tests/test_velocity_tracking.py \
        --load_run 2026-01-22_19-26-20 \
        --checkpoint model_690.pt \
        --step_duration 10.0 \
        --step_values 0.0,0.5,0.0,-0.5

Output:
    - Console: Real-time tracking performance metrics
    - CSV file: logs/velocity_tracking_<timestamp>.csv
    - Metrics: Rise time, settling time, overshoot, steady-state error

CSV Columns:
    time, vx_cmd, vx_actual, wz_cmd, wz_actual, vx_error, wz_error
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Velocity tracking test with response plotting")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-Velocity", help="Task name")

# Policy loading
parser.add_argument("--load_run", type=str, default=None, help="Run directory name for RL policy")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")

# PID control option
parser.add_argument("--use_pid", action="store_true", help="Use PID controller instead of RL policy")
parser.add_argument("--kp_linear", type=float, default=0.10, help="Kp for linear velocity PID")
parser.add_argument("--ki_linear", type=float, default=0.05, help="Ki for linear velocity PID")
parser.add_argument("--kd_linear", type=float, default=0.02, help="Kd for linear velocity PID")
parser.add_argument("--kp_angular", type=float, default=2.0, help="Kp for angular velocity PID")
parser.add_argument("--ki_angular", type=float, default=0.1, help="Ki for angular velocity PID")
parser.add_argument("--kd_angular", type=float, default=0.05, help="Kd for angular velocity PID")

# Test sequence parameters
parser.add_argument("--step_duration", type=float, default=5.0, help="Duration of each step command (seconds)")
parser.add_argument(
    "--step_values",
    type=str,
    default="0.0,0.5,0.0,-0.3,0.0,0.8,0.0",
    help="Comma-separated velocity step values (m/s for linear, rad/s for angular)",
)
parser.add_argument("--test_linear", action="store_true", default=True, help="Test linear velocity (vx)")
parser.add_argument("--test_angular", action="store_true", help="Test angular velocity (wz)")

# Logging parameters
parser.add_argument("--log_interval", type=int, default=1, help="Log every N steps")

# Control parameters
parser.add_argument("--wheel_base", type=float, default=0.2, help="Distance between wheels (m)")
parser.add_argument("--wheel_radius", type=float, default=0.05, help="Wheel radius (m)")
parser.add_argument("--effort_scale", type=float, default=100.0, help="Scale factor for wheel effort")
parser.add_argument("--pid_decimation", type=int, default=4, help="PID decimation factor")

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

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


class VelocityPIDController:
    """PID controller for velocity tracking."""

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

        # PID state
        self.error_integral_linear = torch.zeros(num_envs, device=device)
        self.error_integral_angular = torch.zeros(num_envs, device=device)
        self.error_prev_linear = torch.zeros(num_envs, device=device)
        self.error_prev_angular = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset PID state."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.error_integral_linear[env_ids] = 0.0
        self.error_integral_angular[env_ids] = 0.0
        self.error_prev_linear[env_ids] = 0.0
        self.error_prev_angular[env_ids] = 0.0

    def compute(self, vel_cmd: torch.Tensor, vel_current: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute wheel efforts from velocity commands."""
        # Extract velocities
        vx_cmd = vel_cmd[:, 0]
        wz_cmd = vel_cmd[:, 1]
        vx_current = vel_current[:, 0]
        wz_current = vel_current[:, 1]

        # Compute errors
        error_linear = vx_cmd - vx_current
        error_angular = wz_cmd - wz_current

        # Update integral with anti-windup
        self.error_integral_linear += error_linear * dt
        self.error_integral_linear = torch.clamp(self.error_integral_linear, -5.0, 5.0)
        self.error_integral_angular += error_angular * dt
        self.error_integral_angular = torch.clamp(self.error_integral_angular, -5.0, 5.0)

        # Compute derivative
        error_derivative_linear = (error_linear - self.error_prev_linear) / dt
        error_derivative_angular = (error_angular - self.error_prev_angular) / dt

        # PID output
        u_linear = (
            self.kp_linear * error_linear
            + self.ki_linear * self.error_integral_linear
            + self.kd_linear * error_derivative_linear
        )
        u_angular = (
            self.kp_angular * error_angular
            + self.ki_angular * self.error_integral_angular
            + self.kd_angular * error_derivative_angular
        )

        # Update previous errors
        self.error_prev_linear = error_linear.clone()
        self.error_prev_angular = error_angular.clone()

        # Convert to wheel velocities (differential drive kinematics)
        v_left = u_linear - (self.wheel_base / 2.0) * u_angular
        v_right = u_linear + (self.wheel_base / 2.0) * u_angular

        # Convert to wheel efforts
        effort_left = v_left * self.effort_scale
        effort_right = v_right * self.effort_scale

        # Stack into (num_envs, 2)
        wheel_efforts = torch.stack([effort_left, effort_right], dim=-1)

        return wheel_efforts


class ResponseLogger:
    """CSV logger for velocity tracking data."""

    def __init__(self, log_dir: str = "logs", filename_prefix: str = "velocity_tracking"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")

        # Create CSV file with header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "vx_cmd", "vx_actual", "wz_cmd", "wz_actual", "vx_error", "wz_error"])

        print(f"[INFO] Velocity tracking data will be logged to: {self.filepath}")

    def log(
        self,
        time: float,
        vx_cmd: float,
        vx_actual: float,
        wz_cmd: float,
        wz_actual: float,
    ):
        """Log a single data point to CSV."""
        vx_error = vx_cmd - vx_actual
        wz_error = wz_cmd - wz_actual

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{time:.4f}",
                    f"{vx_cmd:.6f}",
                    f"{vx_actual:.6f}",
                    f"{wz_cmd:.6f}",
                    f"{wz_actual:.6f}",
                    f"{vx_error:.6f}",
                    f"{wz_error:.6f}",
                ]
            )

    def get_filepath(self) -> str:
        return self.filepath


def load_policy(env, agent_cfg, checkpoint_path: str):
    """Load trained RL policy from checkpoint."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return None

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Policy loaded successfully")
    return policy


def compute_tracking_metrics(time_data, cmd_data, actual_data, step_times):
    """Compute tracking performance metrics for each step.

    Metrics:
    - Rise time (10% to 90%)
    - Settling time (within 5% of final value)
    - Overshoot (%)
    - Steady-state error

    Args:
        time_data: Array of time values
        cmd_data: Array of command values
        actual_data: Array of actual values
        step_times: List of step transition times

    Returns:
        Dictionary of metrics
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

        if len(t_step) < 10:
            continue

        # Get command value (should be constant)
        cmd_value = cmd_step[0]

        if abs(cmd_value) < 0.01:
            # Skip zero commands
            continue

        # Rise time (10% to 90%)
        threshold_10 = cmd_value * 0.1
        threshold_90 = cmd_value * 0.9
        idx_10 = np.where(np.abs(actual_step) >= np.abs(threshold_10))[0]
        idx_90 = np.where(np.abs(actual_step) >= np.abs(threshold_90))[0]

        rise_time = None
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = t_step[idx_90[0]] - t_step[idx_10[0]]

        # Settling time (within 5% of final value)
        final_value = cmd_value
        threshold_settle = np.abs(final_value) * 0.05
        settled_mask = np.abs(actual_step - final_value) <= threshold_settle

        settling_time = None
        if np.any(settled_mask):
            # Find first time it settles AND stays settled
            for j in range(len(t_step) - 10):
                if np.all(settled_mask[j : j + 10]):
                    settling_time = t_step[j]
                    break

        # Overshoot
        overshoot = 0.0
        if cmd_value != 0:
            max_value = np.max(np.abs(actual_step))
            if max_value > np.abs(cmd_value):
                overshoot = (max_value - np.abs(cmd_value)) / np.abs(cmd_value) * 100.0

        # Steady-state error (average of last 20% of data)
        steady_idx = int(len(actual_step) * 0.8)
        steady_state_error = np.mean(cmd_step[steady_idx:] - actual_step[steady_idx:])

        metrics.append(
            {
                "step": i,
                "command": cmd_value,
                "rise_time": rise_time,
                "settling_time": settling_time,
                "overshoot": overshoot,
                "steady_state_error": steady_state_error,
            }
        )

    return metrics


def main():
    """Main function."""

    # Parse step values
    step_values = [float(x) for x in args_cli.step_values.split(",")]
    print(f"[INFO] Test sequence: {step_values}")

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Calculate total test duration
    total_duration = len(step_values) * args_cli.step_duration
    env_cfg.episode_length_s = total_duration + 5.0  # Add buffer

    env = gym.make(args_cli.task, cfg=env_cfg)

    print("\n" + "=" * 80)
    print("EVOBOT V1 - VELOCITY TRACKING TEST")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Test duration: {total_duration:.1f} seconds")
    print(f"Step duration: {args_cli.step_duration:.1f} seconds")
    print(f"Step values: {step_values}")
    print("=" * 80)

    # Load controller/policy
    policy = None
    pid_controller = None

    if args_cli.use_pid:
        print("\n[INFO] Using PID controller")
        print(f"Linear PID:  Kp={args_cli.kp_linear}, Ki={args_cli.ki_linear}, Kd={args_cli.kd_linear}")
        print(f"Angular PID: Kp={args_cli.kp_angular}, Ki={args_cli.ki_angular}, Kd={args_cli.kd_angular}")

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
        pid_controller.reset()
    else:
        if args_cli.load_run is None:
            print("[ERROR] Must specify --load_run for RL policy or use --use_pid")
            env.close()
            return

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        env_wrapped = RslRlVecEnvWrapper(env)

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        checkpoint_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)

        policy = load_policy(env_wrapped, agent_cfg, checkpoint_path)
        if policy is None:
            print("[ERROR] Failed to load policy")
            env.close()
            return

    # Initialize logger
    logger = ResponseLogger()

    # Reset environment
    if args_cli.use_pid:
        env.reset()
    else:
        env.reset()
        obs = env_wrapped.get_observations()

    print("[INFO] Starting velocity tracking test...\n")

    # Get robot asset
    robot = env.unwrapped.scene["robot"]

    # Get timestep
    dt_rl = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation
    dt_pid = dt_rl * args_cli.pid_decimation if args_cli.use_pid else dt_rl

    # Data storage
    time_log = []
    vx_cmd_log = []
    vx_actual_log = []
    wz_cmd_log = []
    wz_actual_log = []
    step_times = [0.0]

    # Main loop
    step_count = 0
    pid_step_counter = 0
    elapsed_time = 0.0
    current_step_idx = 0

    # Initial command
    vel_cmd = torch.zeros(2, device=args_cli.device)
    if args_cli.test_linear:
        vel_cmd[0] = step_values[current_step_idx]
    if args_cli.test_angular:
        vel_cmd[1] = step_values[current_step_idx]

    print(f"{'Time':>8s} {'VxCmd':>10s} {'VxAct':>10s} {'WzCmd':>10s} {'WzAct':>10s} {'VxErr':>10s} {'WzErr':>10s}")
    print("-" * 78)

    wheel_efforts = torch.zeros(args_cli.num_envs, 2, device=args_cli.device)

    try:
        while simulation_app.is_running() and elapsed_time < total_duration:
            with torch.inference_mode():
                # Update command based on time
                if elapsed_time >= (current_step_idx + 1) * args_cli.step_duration:
                    current_step_idx += 1
                    if current_step_idx < len(step_values):
                        if args_cli.test_linear:
                            vel_cmd[0] = step_values[current_step_idx]
                        if args_cli.test_angular:
                            vel_cmd[1] = step_values[current_step_idx]
                        step_times.append(elapsed_time)
                        print(f"\n[STEP {current_step_idx}] New command: vx={vel_cmd[0]:.2f}, wz={vel_cmd[1]:.2f}\n")

                # Get current velocity
                vel_current = torch.stack(
                    [
                        robot.data.root_lin_vel_b[:, 0],  # vx
                        robot.data.root_ang_vel_b[:, 2],  # wz
                    ],
                    dim=-1,
                )

                # Get action
                if args_cli.use_pid:
                    if pid_step_counter % args_cli.pid_decimation == 0:
                        vel_cmd_expanded = vel_cmd.unsqueeze(0).expand(args_cli.num_envs, -1)
                        wheel_efforts = pid_controller.compute(vel_cmd_expanded, vel_current, dt_pid)

                    # Apply wheel efforts
                    joint_efforts = torch.zeros(args_cli.num_envs, robot.num_joints, device=args_cli.device)
                    wheel_indices = [
                        robot.joint_names.index("left_wheel_joint"),
                        robot.joint_names.index("right_wheel_joint"),
                    ]
                    joint_efforts[:, wheel_indices[0]] = wheel_efforts[:, 0]
                    joint_efforts[:, wheel_indices[1]] = wheel_efforts[:, 1]
                    robot.set_joint_effort_target(joint_efforts)
                    robot.write_data_to_sim()

                    # Step with zero action (efforts already applied)
                    obs, reward, terminated, truncated, info = env.step(
                        torch.zeros(args_cli.num_envs, env.action_space.shape[0], device=args_cli.device)
                    )

                    pid_step_counter += 1
                else:
                    # Update command in observation (if using RL policy)
                    # Note: Assumes velocity command is in observation
                    actions = policy(obs)
                    # RslRlVecEnvWrapper returns 4 values (obs, reward, done, info) instead of 5
                    obs, reward, _, _ = env_wrapped.step(actions)

                # Log data
                if step_count % args_cli.log_interval == 0:
                    vx_cmd_val = vel_cmd[0].item()
                    wz_cmd_val = vel_cmd[1].item()
                    vx_actual_val = vel_current[0, 0].item()
                    wz_actual_val = vel_current[0, 1].item()

                    time_log.append(elapsed_time)
                    vx_cmd_log.append(vx_cmd_val)
                    vx_actual_log.append(vx_actual_val)
                    wz_cmd_log.append(wz_cmd_val)
                    wz_actual_log.append(wz_actual_val)

                    logger.log(elapsed_time, vx_cmd_val, vx_actual_val, wz_cmd_val, wz_actual_val)

                    # Print every 50 steps
                    if step_count % 50 == 0:
                        vx_err = vx_cmd_val - vx_actual_val
                        wz_err = wz_cmd_val - wz_actual_val
                        print(
                            f"{elapsed_time:8.2f} {vx_cmd_val:10.3f} {vx_actual_val:10.3f} {wz_cmd_val:10.3f} {wz_actual_val:10.3f} {vx_err:10.3f} {wz_err:10.3f}"  # noqa: E501
                        )

                step_count += 1
                elapsed_time += dt_rl

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")

    # Compute metrics
    print("\n" + "=" * 80)
    print("TRACKING PERFORMANCE METRICS")
    print("=" * 80)

    time_array = np.array(time_log)
    vx_cmd_array = np.array(vx_cmd_log)
    vx_actual_array = np.array(vx_actual_log)
    wz_cmd_array = np.array(wz_cmd_log)
    wz_actual_array = np.array(wz_actual_log)

    if args_cli.test_linear:
        print("\nLinear Velocity (vx) Tracking:")
        vx_metrics = compute_tracking_metrics(time_array, vx_cmd_array, vx_actual_array, step_times)
        for m in vx_metrics:
            print(f"  Step {m['step']}: cmd={m['command']:.3f} m/s")
            if m["rise_time"]:
                print(f"    Rise time: {m['rise_time']:.3f} s")
            if m["settling_time"]:
                print(f"    Settling time: {m['settling_time']:.3f} s")
            print(f"    Overshoot: {m['overshoot']:.2f}%")
            print(f"    Steady-state error: {m['steady_state_error']:.4f} m/s")

    if args_cli.test_angular:
        print("\nAngular Velocity (wz) Tracking:")
        wz_metrics = compute_tracking_metrics(time_array, wz_cmd_array, wz_actual_array, step_times)
        for m in wz_metrics:
            print(f"  Step {m['step']}: cmd={m['command']:.3f} rad/s")
            if m["rise_time"]:
                print(f"    Rise time: {m['rise_time']:.3f} s")
            if m["settling_time"]:
                print(f"    Settling time: {m['settling_time']:.3f} s")
            print(f"    Overshoot: {m['overshoot']:.2f}%")
            print(f"    Steady-state error: {m['steady_state_error']:.4f} rad/s")

    print("=" * 80)

    # Generate plotting script
    print("\n" + "=" * 80)
    print("PLOT DATA WITH PYTHON:")
    print("=" * 80)
    print(f"""
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('{logger.get_filepath()}')

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Linear velocity tracking
axes[0].plot(df['time'], df['vx_cmd'], 'r--', linewidth=2, label='Command')
axes[0].plot(df['time'], df['vx_actual'], 'b-', linewidth=1.5, label='Actual')
axes[0].set_xlabel('Time (s)', fontsize=12)
axes[0].set_ylabel('Linear Velocity (m/s)', fontsize=12)
axes[0].set_title('Linear Velocity Tracking', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Angular velocity tracking
axes[1].plot(df['time'], df['wz_cmd'], 'r--', linewidth=2, label='Command')
axes[1].plot(df['time'], df['wz_actual'], 'b-', linewidth=1.5, label='Actual')
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
axes[1].set_title('Angular Velocity Tracking', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('velocity_tracking.png', dpi=300, bbox_inches='tight')
print("Plot saved to: velocity_tracking.png")
plt.show()
""")
    print("=" * 80)

    # Close environment
    print("\n[INFO] Closing environment...")
    env.close()
    print(f"[INFO] Data saved to: {logger.get_filepath()}")
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
