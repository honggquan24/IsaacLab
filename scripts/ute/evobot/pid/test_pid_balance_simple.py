# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple PID Balance Test - Compare pitch angle for robot balance
Similar to rpy_alignment_imu but simplified for PID testing

Usage:
    ./isaaclab.sh -p scripts/ute/evobot/pid/test_pid_balance_simple.py \
        --num_envs 1 --kp 1.0 --ki 0.0 --kd 0.01 --episode_length 10
"""

import argparse
import contextlib
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Simple PID balance test - pitch angle comparison")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--kp", type=float, default=5.0, help="Proportional gain")
parser.add_argument("--ki", type=float, default=0.1, help="Integral gain")
parser.add_argument("--kd", type=float, default=0.4, help="Derivative gain")
parser.add_argument("--target_pitch", type=float, default=0.0, help="Target pitch angle (rad)")
parser.add_argument("--episode_length", type=float, default=5.0, help="Episode length (s)")
parser.add_argument("--max_steps", type=int, default=60 * 5, help="Max steps")
parser.add_argument("--headless", action="store_true", help="Headless mode")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after Isaac Sim launch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.evobot.navigation.velocity import EvobotVelocityBalanceEnvCfg


class SimplePIDController:
    """Simple PID controller for pitch angle"""

    def __init__(self, kp, ki, kd, target=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_pitch, dt):
        """Compute PID output"""
        # Error (simple angle wrapping without tensor)
        error = current_pitch - self.target

        # Wrap to [-pi, pi]
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi

        # PID terms
        p_term = self.kp * error

        self.integral += error * dt
        i_term = self.ki * self.integral

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        self.prev_error = error

        return output, p_term, i_term, d_term, error


def get_pitch_from_imu(imu_sensor):
    """
    Get pitch angle from IMU quaternion
    Similar to rpy_alignment_imu logic
    """
    # Get quaternion (world frame)
    quat = imu_sensor.data.quat_w

    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    # Convert to Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quat)

    # Clamp pitch to avoid extreme values
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)

    return pitch


def save_data(data_log, output_dir, timestamp, params):
    """Save CSV and plot"""
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    csv_file = os.path.join(output_dir, f"pid_balance_{timestamp}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time",
                "pitch",
                "error",
                "pid_output",
                "p_term",
                "i_term",
                "d_term",
                "left_wheel",
                "right_wheel",
                "height",
            ]
        )
        writer.writerows(data_log)
    print(f"✓ CSV saved: {csv_file}")

    # Plot
    data = np.array(data_log)
    time = data[:, 1]
    pitch = np.rad2deg(data[:, 2])
    error = np.rad2deg(data[:, 3])
    pid_output = data[:, 4]
    height = data[:, 10]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"PID Balance - Pitch Control\nKP={params['kp']}, KI={params['ki']}, KD={params['kd']}",
        fontsize=14,
        fontweight="bold",
    )

    # Pitch angle
    axes[0, 0].plot(time, pitch, "b-", linewidth=2)
    axes[0, 0].axhline(y=np.rad2deg(params["target"]), color="r", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Pitch Angle (deg)")
    axes[0, 0].set_title("Pitch Angle Response")
    axes[0, 0].grid(True, alpha=0.3)

    # Error
    axes[0, 1].plot(time, error, "r-", linewidth=2)
    axes[0, 1].axhline(y=0, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (deg)")
    axes[0, 1].set_title("Pitch Error")
    axes[0, 1].grid(True, alpha=0.3)

    # PID output
    axes[1, 0].plot(time, pid_output, "g-", linewidth=2)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Motor Effort")
    axes[1, 0].set_title("PID Control Signal")
    axes[1, 0].grid(True, alpha=0.3)

    # Height
    axes[1, 1].plot(time, height, "m-", linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Height (m)")
    axes[1, 1].set_title("Robot Height")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"pid_balance_response_{timestamp}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved: {plot_file}")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("SIMPLE PID BALANCE TEST - Pitch Angle Control")
    print("=" * 70)
    print(f"PID Gains: KP={args.kp}, KI={args.ki}, KD={args.kd}")
    print(f"Target Pitch: {args.target_pitch} rad ({np.rad2deg(args.target_pitch):.1f}°)")
    print(f"Episode Length: {args.episode_length}s")
    print("=" * 70 + "\n")

    # Setup environment
    env_cfg = EvobotVelocityBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length

    # Disable all terminations except timeout
    for term_name in dir(env_cfg.terminations):
        if not term_name.startswith("_") and term_name != "time_out":
            with contextlib.suppress(Exception):
                delattr(env_cfg.terminations, term_name)

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Create PID controllers
    pids = [SimplePIDController(args.kp, args.ki, args.kd, args.target_pitch) for _ in range(args.num_envs)]

    dt = env.step_dt
    obs, _ = env.reset()

    data_log = []

    print("Running simulation...\n")
    print(f"{'Step':>5} | {'Time':>6} | {'Pitch':>8} | {'Error':>8} | {'PID':>8} | {'Height':>7}")
    print("-" * 70)

    try:
        for step in range(args.max_steps):
            # Get IMU pitch
            imu = env.scene["imu"]
            robot = env.scene["robot"]

            pitch = get_pitch_from_imu(imu)

            # Compute PID
            wheel_efforts = torch.zeros(args.num_envs, device=env.device)
            log_data = None

            for i in range(args.num_envs):
                output, p, i_term, d, error = pids[i].compute(pitch[i].item(), dt)
                wheel_efforts[i] = output

                if i == 0:  # Log first env
                    height = robot.data.root_pos_w[0, 2].item()
                    log_data = [step, step * dt, pitch[0].item(), error, output, p, i_term, d, output, output, height]

            # Apply to both wheels
            actions = torch.zeros(args.num_envs, 5, device=env.device)
            actions[:, 0] = wheel_efforts  # Left wheel
            actions[:, 1] = wheel_efforts  # Right wheel

            obs, rewards, terminated, truncated, info = env.step(actions)

            if log_data:
                data_log.append(log_data)

            # Print progress
            if step % 50 == 0:
                print(
                    f"{step:5d} | {step * dt:6.2f} | "
                    f"{np.rad2deg(pitch[0].item()):8.2f} | "
                    f"{np.rad2deg(log_data[3]):8.2f} | "
                    f"{log_data[4]:8.1f} | "
                    f"{log_data[10]:7.3f}"
                )

            if terminated.all():
                print(f"\nTerminated at step {step}")
                break

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")

    finally:
        # Save results
        print("\n" + "=" * 70)
        print("Saving results...")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "logs", "pid_manual")

        params = {"kp": args.kp, "ki": args.ki, "kd": args.kd, "target": args.target_pitch}

        save_data(data_log, output_dir, timestamp, params)

        # Metrics
        data = np.array(data_log)
        pitch_values = data[:, 2]
        errors = data[:, 3]

        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS")
        print("=" * 70)
        print(f"Final Pitch: {np.rad2deg(pitch_values[-1]):.2f}°")
        print(f"Final Error: {np.rad2deg(errors[-1]):.2f}°")
        print(f"Mean Abs Error: {np.rad2deg(np.abs(errors).mean()):.2f}°")
        print(f"Max Error: {np.rad2deg(np.abs(errors).max()):.2f}°")
        print("=" * 70 + "\n")

        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
