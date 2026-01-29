#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PID Balance Controller Test - Based on Research Paper
"A Two Wheel Self-Balancing Vehicle" (2021)

This implements the PID balance control from the paper:
- Complementary filter for IMU data fusion (Eq. 19)
- PID controller for tilt angle (KP=52, KI=75, KD=0.3)
- Same effort applied to BOTH wheels for balance

The test saves data to CSV and generates response plots.

Usage:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/pid_tests/test_pid_balance_paper.py \
        --num_envs 1 --kp 1.0 --ki 0.0 --kd 0.01
"""

import argparse
import torch
import numpy as np
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import Isaac Lab
from isaaclab.app import AppLauncher

# Parse arguments BEFORE launching Isaac Sim
parser = argparse.ArgumentParser(description="Test PID balance controller from paper")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--kp", type=float, default=1.0, help="Proportional gain (paper: 52)")
parser.add_argument("--ki", type=float, default=0.0, help="Integral gain (paper: 75)")
parser.add_argument("--kd", type=float, default=0.01, help="Derivative gain (paper: 0.3)")
parser.add_argument("--setpoint", type=float, default=0.0, help="Target angle in radians (0 = upright)")
parser.add_argument("--alpha", type=float, default=0.98, help="Complementary filter coefficient (paper: 0.98)")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum simulation steps")
parser.add_argument("--episode_length", type=float, default=10.0, help="Episode length in seconds (default: 10s)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--keep_open", action="store_true", default=True, help="Keep simulation open after test")
parser.add_argument("--replay_steps", type=int, default=500, help="Steps to continue running after test")
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching Isaac Sim
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from isaaclab_assets.evobot_v1.navigation.velocity import EvobotV1VelocityBalanceEnvCfg


@configclass
class DummyTerminationsCfg:
    """Dummy terminations config with ONLY time_out for PID testing.

    This prevents robot from terminating early due to:
    - Height drops
    - Bad orientation
    - High joint velocities
    - Contact violations

    Allowing full episode duration to observe PID response.
    """
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )


class ComplementaryFilter:
    """
    Complementary Filter from paper (Eq. 19)

    filtered_θ = α(previous_θ + gyroAngVelY*dt) + (1-α)(accAngle)

    - High-pass filter on gyroscope (removes drift)
    - Low-pass filter on accelerometer (removes vibration noise)
    """
    def __init__(self, alpha=0.98, dt=0.02):
        self.alpha = alpha
        self.dt = dt
        self.filtered_angle = 0.0

    def update(self, gyro_rate, acc_angle):
        """
        Args:
            gyro_rate: Angular velocity from gyroscope (rad/s)
            acc_angle: Angle from accelerometer (rad)

        Returns:
            filtered_angle: Fused angle estimate (rad)
        """
        # High-pass (gyro) + Low-pass (acc)
        self.filtered_angle = self.alpha * (self.filtered_angle + gyro_rate * self.dt) + \
                             (1 - self.alpha) * acc_angle
        return self.filtered_angle

    def reset(self):
        """Reset filter state"""
        self.filtered_angle = 0.0


class BalancePIDController:
    """
    PID Controller for balance control from paper

    Paper parameters (manual tuning using Ziegler-Nichols):
    - KP = 1.0
    - KI = 0.0
    - KD = 0.01
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.01, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint  # Target angle (0 = upright)

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, filtered_angle, dt):
        """
        Compute PID output based on filtered tilt angle

        Args:
            filtered_angle: Current tilt angle from complementary filter (rad)
            dt: Time step (s)

        Returns:
            output: Motor effort/torque for BOTH wheels (same value)
        """
        # Error (paper: e(t) = filtered_θ - SetPoint)
        error = filtered_angle - self.setpoint

        # PID terms
        p_term = self.kp * error

        self.integral += error * dt
        i_term = self.ki * self.integral

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Total output
        output = p_term + i_term + d_term

        self.prev_error = error

        return output, p_term, i_term, d_term

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0


def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)

    Args:
        quat: Quaternion [x, y, z, w] tensor of shape [num_envs, 4]

    Returns:
        roll, pitch, yaw: Euler angles in radians
    """
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    # Roll (rotation about x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Yaw (rotation about z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_acc_angle_from_imu(imu_sensor):
    """
    Calculate tilt angle from accelerometer (paper Eq. 20)

    accAngle = accX / sqrt(accX^2 + accZ^2)

    Args:
        imu_sensor: IMU sensor from scene

    Returns:
        acc_angle: Tilt angle from accelerometer (rad)
    """
    # Get linear acceleration in body frame
    acc = imu_sensor.data.lin_acc_b  # [num_envs, 3]
    acc_x = acc[:, 0]
    acc_z = acc[:, 2]

    # Compute angle using atan2 for proper quadrant
    acc_angle = torch.atan2(acc_x, torch.sqrt(acc_x**2 + acc_z**2))

    return acc_angle


def save_to_csv(data_log, filename):
    """Save logged data to CSV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow([
            'step', 'time', 'gyro_rate', 'acc_angle', 'filtered_angle',
            'pid_output', 'p_term', 'i_term', 'd_term',
            'left_wheel_effort', 'right_wheel_effort',
            'base_height', 'base_vel_x'
        ])
        # Data
        writer.writerows(data_log)

    print(f"✓ Data saved to: {filename}")


def plot_response(data_log, filename, params):
    """Generate response plots"""
    data = np.array(data_log)
    time = data[:, 1]
    filtered_angle = data[:, 4]
    pid_output = data[:, 5]
    p_term = data[:, 6]
    i_term = data[:, 7]
    d_term = data[:, 8]
    base_height = data[:, 11]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PID Balance Controller Response\n' +
                 f'KP={params["kp"]}, KI={params["ki"]}, KD={params["kd"]}, ' +
                 f'α={params["alpha"]}, Setpoint={params["setpoint"]}°',
                 fontsize=14, fontweight='bold')

    # Plot 1: Filtered Angle vs Time
    ax1 = axes[0, 0]
    ax1.plot(time, np.rad2deg(filtered_angle), 'b-', linewidth=2, label='Filtered Angle')
    ax1.axhline(y=np.rad2deg(params["setpoint"]), color='r', linestyle='--',
                linewidth=2, label=f'Setpoint ({params["setpoint"]}°)')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Tilt Angle (degrees)', fontsize=11)
    ax1.set_title('Tilt Angle Response', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: PID Output (Motor Effort)
    ax2 = axes[0, 1]
    ax2.plot(time, pid_output, 'g-', linewidth=2, label='PID Output')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Motor Effort (N⋅m)', fontsize=11)
    ax2.set_title('PID Control Signal (Both Wheels)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: PID Terms Breakdown
    ax3 = axes[1, 0]
    ax3.plot(time, p_term, 'r-', linewidth=1.5, label='P term', alpha=0.8)
    ax3.plot(time, i_term, 'g-', linewidth=1.5, label='I term', alpha=0.8)
    ax3.plot(time, d_term, 'b-', linewidth=1.5, label='D term', alpha=0.8)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Term Value', fontsize=11)
    ax3.set_title('PID Terms Breakdown', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Base Height (Stability indicator)
    ax4 = axes[1, 1]
    ax4.plot(time, base_height, 'm-', linewidth=2, label='Base Height')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Height (m)', fontsize=11)
    ax4.set_title('Robot Base Height (Stability)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {filename}")
    plt.close()


def main():
    """Main test function"""

    print("\n" + "="*70)
    print("PID BALANCE CONTROLLER TEST - Based on Research Paper")
    print("="*70)
    print(f"Paper: 'A Two Wheel Self-Balancing Vehicle' (2021)")
    print(f"Authors: Ďuriš et al., Slovak University of Technology")
    print("="*70)
    print(f"PID Gains: KP={args.kp}, KI={args.ki}, KD={args.kd}")
    print(f"Complementary Filter: α={args.alpha}")
    print(f"Target Angle (Setpoint): {args.setpoint} rad ({np.rad2deg(args.setpoint):.1f}°)")
    print(f"Number of Environments: {args.num_envs}")
    print(f"Max Steps: {args.max_steps}")
    print("="*70 + "\n")

    # Create environment
    env_cfg = EvobotV1VelocityBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda"

    # IMPORTANT: Increase episode length for PID testing
    # Default is 10s which is too short to observe PID response
    env_cfg.episode_length_s = args.episode_length  # User-specified or default 60s

    # CRITICAL: Disable all terminations except time_out
    # This prevents early termination from height/orientation/velocity violations
    # We need to keep the original TerminationsCfg structure but disable terms
    original_terminations = env_cfg.terminations

    # Disable all terminations except time_out
    for term_name in dir(original_terminations):
        if not term_name.startswith('_') and term_name != 'time_out':
            try:
                delattr(original_terminations, term_name)
            except:
                pass

    print("⚠️  Disabled all terminations except time_out - robot won't terminate from falling/orientation")

    print(f"Episode length: {env_cfg.episode_length_s}s")
    print(f"Decimation: {env_cfg.decimation}")
    print(f"Sim dt: {env_cfg.sim.dt}s")
    print(f"Expected steps per episode: {env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation):.0f}")
    print()

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Create complementary filters (one per environment)
    num_envs = args.num_envs
    dt = env.step_dt
    filters = [ComplementaryFilter(alpha=args.alpha, dt=dt) for _ in range(num_envs)]

    # Create PID controllers (one per environment)
    pids = [BalancePIDController(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        setpoint=args.setpoint
    ) for _ in range(num_envs)]

    # Reset environment
    obs, _ = env.reset()

    # Data logging (only log first environment for clarity)
    data_log = []

    print("Running simulation...\n")
    print(f"{'Step':>5} | {'Time':>6} | {'Gyro':>8} | {'Acc':>8} | {'Filter':>8} | "
          f"{'PID':>8} | {'Height':>7}")
    print("-" * 70)

    try:
        for step in range(args.max_steps):
            # Get IMU data
            imu_sensor = env.scene["imu"]
            robot = env.scene["robot"]

            # 1. Get gyroscope angular velocity (pitch rate)
            gyro_rate_y = imu_sensor.data.ang_vel_b[:, 1]  # [num_envs]

            # 2. Get accelerometer angle
            acc_angle = get_acc_angle_from_imu(imu_sensor)  # [num_envs]

            # 3. Apply complementary filter
            filtered_angles = torch.zeros(num_envs, device=env.device)
            for i in range(num_envs):
                filtered_angles[i] = filters[i].update(
                    gyro_rate_y[i].item(),
                    acc_angle[i].item()
                )

            # 4. Compute PID output
            wheel_efforts = torch.zeros(num_envs, device=env.device)
            pid_outputs = []
            for i in range(num_envs):
                output, p_term, i_term, d_term = pids[i].compute(
                    filtered_angles[i].item(),
                    dt
                )
                wheel_efforts[i] = output
                if i == 0:  # Store for logging
                    pid_outputs = [output, p_term, i_term, d_term]

            # 5. Apply SAME effort to BOTH wheels (balance control from paper)
            actions = torch.zeros(num_envs, 5, device=env.device)
            actions[:, 0] = wheel_efforts  # Left wheel
            actions[:, 1] = wheel_efforts  # Right wheel (SAME as left)
            # actions[:, 2:5] = 0  # Arm and grippers stationary

            # 6. Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # 7. Log data (first environment only)
            if num_envs > 0:
                current_time = step * dt
                base_height = robot.data.root_pos_w[0, 2].item()
                base_vel_x = robot.data.root_lin_vel_w[0, 0].item()

                data_log.append([
                    step,
                    current_time,
                    gyro_rate_y[0].item(),
                    acc_angle[0].item(),
                    filtered_angles[0].item(),
                    pid_outputs[0],
                    pid_outputs[1],  # P term
                    pid_outputs[2],  # I term
                    pid_outputs[3],  # D term
                    wheel_efforts[0].item(),
                    wheel_efforts[0].item(),  # Same as left wheel
                    base_height,
                    base_vel_x
                ])

            # Print progress
            if step % 50 == 0:
                print(f"{step:5d} | {current_time:6.2f} | "
                      f"{gyro_rate_y[0].item():8.3f} | {np.rad2deg(acc_angle[0].item()):8.2f} | "
                      f"{np.rad2deg(filtered_angles[0].item()):8.2f} | "
                      f"{wheel_efforts[0].item():8.1f} | {base_height:7.3f}")

            # Check termination
            if terminated.any():
                term_count = terminated.sum().item()
                print(f"\n⚠ {term_count} environment(s) TERMINATED at step {step} (time={current_time:.2f}s)")

                # Print termination reasons
                robot = env.scene["robot"]
                for i in range(num_envs):
                    if terminated[i]:
                        height = robot.data.root_pos_w[i, 2].item()
                        angle = np.rad2deg(filtered_angles[i].item())
                        print(f"  Env {i}: Height={height:.3f}m, Angle={angle:.1f}°")

                print("  Possible reasons:")
                print(f"    - Height < 0.25m: {(robot.data.root_pos_w[:, 2] < 0.25).any().item()}")
                print(f"    - Bad orientation (>{150}°): Check orientation")
                print(f"    - Time out ({env_cfg.episode_length_s}s)")
                print()

                # Don't break - let it continue if not all terminated
                if terminated.all():
                    print("  All environments terminated. Stopping test.")
                    break

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")

    finally:
        # Save data
        print("\n" + "="*70)
        print("Saving results...")
        print("="*70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get the script directory and create output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "logs", "pid_manual")
        os.makedirs(output_dir, exist_ok=True)

        # Save CSV
        csv_filename = os.path.join(output_dir, f"pid_balance_{timestamp}.csv")
        save_to_csv(data_log, csv_filename)

        # Plot results
        plot_filename = os.path.join(output_dir, f"pid_balance_response_{timestamp}.png")
        params = {
            "kp": args.kp,
            "ki": args.ki,
            "kd": args.kd,
            "alpha": args.alpha,
            "setpoint": args.setpoint
        }
        plot_response(data_log, plot_filename, params)

        # Performance metrics
        data = np.array(data_log)
        filtered_angles = data[:, 4]
        settling_time = None
        steady_state_error = np.abs(filtered_angles[-50:].mean() - args.setpoint)

        # Find settling time (within 5% of setpoint)
        threshold = 0.05  # 5% of setpoint (in radians)
        for i, angle in enumerate(filtered_angles):
            if np.abs(angle - args.setpoint) < threshold:
                settling_time = data[i, 1]
                break

        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        print(f"Settling Time (±5%): {settling_time:.2f} s" if settling_time else "Not settled")
        print(f"Steady-State Error: {np.rad2deg(steady_state_error):.2f}°")
        print(f"Final Angle: {np.rad2deg(filtered_angles[-1]):.2f}°")
        print(f"Max Overshoot: {np.rad2deg(filtered_angles.max() - args.setpoint):.2f}°")
        print("="*70 + "\n")

        # Keep simulation open for visualization
        if args.keep_open and not args.headless:
            print("\n" + "="*70)
            print("CONTINUING SIMULATION FOR VISUALIZATION")
            print("="*70)
            print(f"Running {args.replay_steps} more steps to observe behavior...")
            print("Press Ctrl+C to stop early\n")

            try:
                for step in range(args.replay_steps):
                    # Continue with same PID control
                    imu_sensor = env.scene["imu"]
                    robot = env.scene["robot"]

                    gyro_rate_y = imu_sensor.data.ang_vel_b[:, 1]
                    acc_angle = get_acc_angle_from_imu(imu_sensor)

                    # Apply complementary filter
                    filtered_angles = torch.zeros(num_envs, device=env.device)
                    for i in range(num_envs):
                        filtered_angles[i] = filters[i].update(
                            gyro_rate_y[i].item(),
                            acc_angle[i].item()
                        )

                    # Compute PID output
                    wheel_efforts = torch.zeros(num_envs, device=env.device)
                    for i in range(num_envs):
                        output, _, _, _ = pids[i].compute(filtered_angles[i].item(), dt)
                        wheel_efforts[i] = output

                    # Apply to both wheels
                    actions = torch.zeros(num_envs, 5, device=env.device)
                    actions[:, 0] = wheel_efforts
                    actions[:, 1] = wheel_efforts

                    # Step
                    obs, rewards, terminated, truncated, info = env.step(actions)

                    # Print periodic updates
                    if step % 50 == 0:
                        print(f"Replay step {step}/{args.replay_steps}: " +
                              f"Angle = {np.rad2deg(filtered_angles[0].item()):.2f}°, " +
                              f"Effort = {wheel_efforts[0].item():.1f}")

                    if terminated.any():
                        print(f"\n⚠ Terminated at replay step {step}")
                        break

            except KeyboardInterrupt:
                print("\n⚠ Visualization interrupted by user")

            print("\n" + "="*70)
            print("Press Enter to close simulation or Ctrl+C to keep it running...")
            print("="*70)
            try:
                input()
            except KeyboardInterrupt:
                print("\n⚠ Keeping simulation open. Close manually when done.")
                # Keep running indefinitely until manual close
                try:
                    while True:
                        # Continue PID control
                        imu_sensor = env.scene["imu"]
                        gyro_rate_y = imu_sensor.data.ang_vel_b[:, 1]
                        acc_angle = get_acc_angle_from_imu(imu_sensor)

                        filtered_angles = torch.zeros(num_envs, device=env.device)
                        for i in range(num_envs):
                            filtered_angles[i] = filters[i].update(
                                gyro_rate_y[i].item(),
                                acc_angle[i].item()
                            )

                        wheel_efforts = torch.zeros(num_envs, device=env.device)
                        for i in range(num_envs):
                            output, _, _, _ = pids[i].compute(filtered_angles[i].item(), dt)
                            wheel_efforts[i] = output

                        actions = torch.zeros(num_envs, 5, device=env.device)
                        actions[:, 0] = wheel_efforts
                        actions[:, 1] = wheel_efforts

                        env.step(actions)
                except KeyboardInterrupt:
                    print("\n\n⚠ Force closing...")

        # Close
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
