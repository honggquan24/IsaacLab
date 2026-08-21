# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PID-only balance control with response data logging.

This script demonstrates pure PID control for Evobot V1 balance task.
No RL policy is used - only PID controller to maintain upright balance.
Response data is logged to CSV for plotting.

Usage:
    # Run with default PID gains
    ./isaaclab.sh -p scripts/ute/evobot/pid/test_pid_balance_response.py

    # Run with custom PID gains
    ./isaaclab.sh -p scripts/ute/evobot/pid/test_pid_balance_response.py \
        --kp_roll 50.0 --ki_roll 0.5 --kd_roll 10.0 \
        --kp_pitch 50.0 --ki_pitch 0.5 --kd_pitch 10.0

    # Run with disturbance test
    ./isaaclab.sh -p scripts/ute/evobot/pid/test_pid_balance_response.py \
        --test_disturbance

Output:
    - Console: Real-time roll/pitch angles and PID outputs
    - CSV file: logs/pid_balance_response_<timestamp>.csv

CSV Columns:
    time, roll, pitch, yaw, roll_error, pitch_error,
    wheel_left_effort, wheel_right_effort, arm_effort
"""

import argparse
import csv
import os
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="PID-only balance control with response logging")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-Balance", help="Task name")

# PID gains for roll (lateral balance)
# Conservative gains to avoid saturation - start low and increase gradually
# Target: smooth control within [-1, 1] range, minimal overshoot
parser.add_argument("--kp_roll", type=float, default=0.0, help="Kp for roll control")
parser.add_argument("--ki_roll", type=float, default=0.0, help="Ki for roll control")
parser.add_argument("--kd_roll", type=float, default=0.0, help="Kd for roll control")

# PID gains for pitch (forward/backward balance)
parser.add_argument("--kp_pitch", type=float, default=0.9, help="Kp for pitch control")
parser.add_argument("--ki_pitch", type=float, default=0.0, help="Ki for pitch control")
parser.add_argument("--kd_pitch", type=float, default=0.02, help="Kd for pitch control")

# Control parameters
parser.add_argument("--effort_scale", type=float, default=1.0, help="Scale factor for wheel effort")
parser.add_argument("--episode_length", type=float, default=30.0, help="Episode length in seconds")
parser.add_argument("--log_interval", type=int, default=1, help="Log every N steps")

# Test settings
parser.add_argument("--test_disturbance", action="store_true", help="Apply random disturbance at t=5s")
parser.add_argument("--disturbance_force", type=float, default=50.0, help="Disturbance force magnitude (N)")

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

from isaaclab.utils.math import euler_xyz_from_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class BalancePIDController:
    """PID controller for 2-wheeled robot balance control.

    Controls roll and pitch angles by commanding wheel and arm efforts.
    Uses separate PID controllers for roll and pitch.
    """

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
        """Initialize PID controller.

        Args:
            kp_roll: Proportional gain for roll
            ki_roll: Integral gain for roll
            kd_roll: Derivative gain for roll
            kp_pitch: Proportional gain for pitch
            ki_pitch: Integral gain for pitch
            kd_pitch: Derivative gain for pitch
            effort_scale: Output scaling factor
            num_envs: Number of parallel environments
            device: Device for tensor operations
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

    def reset(self, env_ids: torch.Tensor = None):
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
        """Compute control efforts from orientation error.

        Args:
            current_rpy: Current roll-pitch-yaw angles (num_envs, 3)
            target_rpy: Target roll-pitch-yaw angles (num_envs, 3)
            dt: Time step (s)

        Returns:
            Control actions [left_wheel, right_wheel, arm, left_grip, right_grip] (num_envs, 5)
        """
        # Extract angles
        roll_current = current_rpy[:, 0]
        pitch_current = current_rpy[:, 1]
        roll_target = target_rpy[:, 0]
        pitch_target = target_rpy[:, 1]

        # Compute errors
        roll_error = roll_target - roll_current
        pitch_error = pitch_target - pitch_current

        # === Roll PID ===
        self.roll_integral += roll_error * dt
        # Anti-windup: clamp integral to prevent excessive buildup
        self.roll_integral = torch.clamp(self.roll_integral, -10.0, 10.0)
        roll_derivative = (roll_error - self.roll_prev_error) / dt
        roll_output = self.kp_roll * roll_error + self.ki_roll * self.roll_integral + self.kd_roll * roll_derivative
        self.roll_prev_error = roll_error.clone()

        # === Pitch PID ===
        self.pitch_integral += pitch_error * dt
        # Anti-windup: clamp integral to prevent excessive buildup
        self.pitch_integral = torch.clamp(self.pitch_integral, -10.0, 10.0)
        pitch_derivative = (pitch_error - self.pitch_prev_error) / dt
        pitch_output = (
            self.kp_pitch * pitch_error + self.ki_pitch * self.pitch_integral + self.kd_pitch * pitch_derivative
        )
        self.pitch_prev_error = pitch_error.clone()

        # Convert to actuator commands
        # Roll control: differential wheel torque (left - right)
        # Pitch control: common mode wheel torque (left + right)
        # Note: effort_scale allows fine-tuning of output magnitude
        wheel_left = (pitch_output + roll_output) * self.effort_scale
        wheel_right = (pitch_output - roll_output) * self.effort_scale

        # Arm control: counter pitch motion (simple feedforward)
        # Disabled (multiplied by 0) - arm not used for balance in this controller
        arm_effort = -pitch_output * 0.0 * self.effort_scale

        # Gripper: keep closed (zero effort = maintain position)
        gripper_left = torch.zeros(self.num_envs, device=self.device)
        gripper_right = torch.zeros(self.num_envs, device=self.device)

        # Stack into action vector [left_wheel, right_wheel, arm, left_grip, right_grip]
        actions = torch.stack([wheel_left, wheel_right, arm_effort, gripper_left, gripper_right], dim=-1)

        # Clamp actions to [-1, 1] range (Isaac Lab action space requirement)
        # If PID gains are tuned well, output should rarely saturate
        actions = torch.clamp(actions, -1.0, 1.0)

        return actions


class ResponseLogger:
    """CSV logger for response data."""

    def __init__(self, log_dir: str = "logs", filename_prefix: str = "pid_balance_response"):
        """Initialize logger.

        Args:
            log_dir: Directory to save logs
            filename_prefix: Prefix for log filename
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")

        # Create CSV file with header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "roll",
                    "pitch",
                    "yaw",
                    "roll_error",
                    "pitch_error",
                    "wheel_left_effort",
                    "wheel_right_effort",
                    "arm_effort",
                ]
            )

        print(f"[INFO] Response data will be logged to: {self.filepath}")

    def log(
        self,
        time: float,
        roll: float,
        pitch: float,
        yaw: float,
        roll_error: float,
        pitch_error: float,
        wheel_left: float,
        wheel_right: float,
        arm_effort: float,
    ):
        """Log a single data point to CSV."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{time:.4f}",
                    f"{roll:.6f}",
                    f"{pitch:.6f}",
                    f"{yaw:.6f}",
                    f"{roll_error:.6f}",
                    f"{pitch_error:.6f}",
                    f"{wheel_left:.6f}",
                    f"{wheel_right:.6f}",
                    f"{arm_effort:.6f}",
                ]
            )


def print_header():
    """Print header information."""
    print("\n" + "=" * 80)
    print("EVOBOT V1 - PID BALANCE CONTROL (RESPONSE LOGGING)")
    print("=" * 80)
    print("Control Strategy:")
    print("  - Roll control: Differential wheel torque (left - right)")
    print("  - Pitch control: Common mode wheel torque (left + right) + arm")
    print("  - Target: Maintain upright vertical orientation (roll=0, pitch=0)")
    print("\nData Logging:")
    print("  - Real-time response data saved to CSV file")
    print("  - Use CSV data to plot step response, settling time, overshoot, etc.")
    print("=" * 80 + "\n")


def main():
    """Main function."""

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Override episode length
    env_cfg.episode_length_s = args_cli.episode_length

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print environment info
    print_header()
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Episode length: {env_cfg.episode_length_s:.1f} seconds")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Print PID configuration
    print("\n" + "=" * 80)
    print("PID CONFIGURATION")
    print("=" * 80)
    print(f"Roll  PID: Kp={args_cli.kp_roll:6.2f}, Ki={args_cli.ki_roll:6.2f}, Kd={args_cli.kd_roll:6.2f}")
    print(f"Pitch PID: Kp={args_cli.kp_pitch:6.2f}, Ki={args_cli.ki_pitch:6.2f}, Kd={args_cli.kd_pitch:6.2f}")
    print(f"Effort scale: {args_cli.effort_scale:.2f}")
    print("=" * 80 + "\n")

    # Initialize PID controller
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

    # Initialize response logger
    logger = ResponseLogger()

    # Reset environment
    obs, _ = env.reset()
    print("[INFO] Environment ready. Starting PID balance control...\n")

    # Get robot asset
    robot = env.unwrapped.scene["robot"]

    # Get timestep
    dt = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation
    control_freq = 1.0 / dt
    print(f"[INFO] Control Frequency: {control_freq:.1f} Hz (dt={dt:.4f}s)\n")

    # Target orientation (upright)
    target_rpy = torch.zeros(args_cli.num_envs, 3, device=args_cli.device)

    # Disturbance flag
    disturbance_applied = False
    disturbance_time = 5.0  # Apply disturbance at t=5s

    # Main loop
    step_count = 0
    elapsed_time = 0.0

    print("[INFO] Starting control loop. Press Ctrl+C to stop.\n")
    print(
        f"{'Time':>8s} {'Roll':>10s} {'Pitch':>10s} {'Yaw':>10s} {'R_Error':>10s} {'P_Error':>10s} {'WheelL':>10s} {'WheelR':>10s} {'Arm':>10s}"  # noqa: E501
    )
    print("-" * 108)

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Get current orientation from quaternion
                quat = robot.data.root_quat_w  # (num_envs, 4) - [w, x, y, z]
                roll, pitch, yaw = euler_xyz_from_quat(quat)  # Returns tuple of (roll, pitch, yaw) tensors
                current_rpy = torch.stack([roll, pitch, yaw], dim=-1)  # Stack to (num_envs, 3)

                # Compute PID control
                actions = pid_controller.compute(current_rpy, target_rpy, dt)

                # Apply disturbance if enabled
                if args_cli.test_disturbance and not disturbance_applied and elapsed_time >= disturbance_time:
                    print(f"\n[DISTURBANCE] Applying force impulse at t={elapsed_time:.2f}s")
                    # Apply lateral force (roll disturbance)
                    force = torch.zeros(args_cli.num_envs, 3, device=args_cli.device)
                    force[:, 1] = args_cli.disturbance_force  # Y-axis force
                    robot.set_external_force_and_torque(force, torch.zeros_like(force))
                    robot.write_data_to_sim()
                    disturbance_applied = True

                # Step environment
                obs, reward, terminated, truncated, info = env.step(actions)
                dones = terminated | truncated

                # Log data (only for first environment)
                if step_count % args_cli.log_interval == 0:
                    roll = current_rpy[0, 0].item()
                    pitch = current_rpy[0, 1].item()
                    yaw = current_rpy[0, 2].item()
                    roll_error = (target_rpy[0, 0] - current_rpy[0, 0]).item()
                    pitch_error = (target_rpy[0, 1] - current_rpy[0, 1]).item()
                    wheel_left = actions[0, 0].item()
                    wheel_right = actions[0, 1].item()
                    arm_effort = actions[0, 2].item()

                    # Log to CSV
                    logger.log(
                        elapsed_time,
                        roll,
                        pitch,
                        yaw,
                        roll_error,
                        pitch_error,
                        wheel_left,
                        wheel_right,
                        arm_effort,
                    )

                    # Print to console (every 10 log intervals for readability)
                    if step_count % (args_cli.log_interval * 10) == 0:
                        print(
                            f"{elapsed_time:8.2f} {roll:10.4f} {pitch:10.4f} {yaw:10.4f} {roll_error:10.4f} {pitch_error:10.4f} {wheel_left:10.4f} {wheel_right:10.4f} {arm_effort:10.4f}"  # noqa: E501
                        )

                # Handle resets
                if dones.any():
                    print(f"\n[RESET] Episode terminated at t={elapsed_time:.2f}s")
                    reset_ids = torch.where(dones)[0]
                    pid_controller.reset(reset_ids)
                    disturbance_applied = False

                step_count += 1
                elapsed_time += dt

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user. Shutting down...")

    # Close environment
    print("\n[INFO] Closing environment...")
    env.close()
    print(f"[INFO] Response data saved to: {logger.filepath}")
    print("[INFO] Done!")


if __name__ == "__main__":
    # Run main
    main()
    # Close sim app
    simulation_app.close()
