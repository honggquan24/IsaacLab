#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Auto-Tune PID for Balance Control - Like Training AI

This script automatically tunes PID gains using optimization algorithm.
User only needs to specify number of epochs (iterations).

Usage:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/pid_tests/auto_tune_pid.py \
        --epochs 20 --headless
"""

import argparse
import torch
import numpy as np
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Auto-tune PID gains like training AI")
parser.add_argument("--epochs", type=int, default=20, help="Number of tuning iterations")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--episode_length", type=float, default=5.0, help="Episode length per trial (s)")
parser.add_argument("--max_steps", type=int, default=250, help="Max steps per trial")
parser.add_argument("--headless", action="store_true", help="Headless mode")
parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate for gain adjustment")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after Isaac Sim launch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat
import gymnasium as gym

# Import environment registry
import isaaclab_assets  # This registers all evobot environments


class SimplePIDController:
    """Simple PID controller"""
    def __init__(self, kp, ki, kd, target=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_pitch, dt):
        """Compute PID output"""
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

        return output, error

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0


def get_pitch_from_imu(imu_sensor):
    """Get pitch angle from IMU"""
    quat = imu_sensor.data.quat_w
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)
    return pitch


def evaluate_pid(env, kp, ki, kd, max_steps, dt):
    """
    Evaluate PID performance
    Returns: cost (lower is better), survived (did not fall)
    """
    # Create controller
    pid = SimplePIDController(kp, ki, kd, target=0.0)

    # Reset environment
    obs, _ = env.reset()

    # Tracking metrics
    total_error = 0.0
    total_effort = 0.0
    max_error = 0.0
    survived = True
    survival_time = 0.0

    imu = env.scene["imu"]
    robot = env.scene["robot"]

    for step in range(max_steps):
        # Get pitch
        pitch = get_pitch_from_imu(imu)

        # Compute PID
        output, error = pid.compute(pitch[0].item(), dt)

        # Apply to wheels
        actions = torch.zeros(args.num_envs, 5, device=env.device)
        actions[:, 0] = output  # Left wheel
        actions[:, 1] = output  # Right wheel

        # Step
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Check if robot fell (height < 0.25m)
        height = robot.data.root_pos_w[0, 2].item()
        if height < 0.25:
            survived = False
            survival_time = step * dt
            break

        # Accumulate metrics
        total_error += abs(error)
        total_effort += abs(output)
        max_error = max(max_error, abs(error))
        survival_time = (step + 1) * dt

    # Calculate cost (lower is better)
    # Cost = weighted sum of error, effort, and penalty for falling
    avg_error = total_error / max(step + 1, 1)
    avg_effort = total_effort / max(step + 1, 1)

    if not survived:
        # Huge penalty for falling
        cost = 1000.0 + avg_error * 100 + avg_effort
    else:
        # Normal cost: balance error and control effort
        cost = avg_error * 10.0 + avg_effort * 0.1 + max_error * 5.0

    return cost, survived, survival_time, avg_error, max_error


def main():
    print("\n" + "="*70)
    print("AUTO-TUNE PID - Like Training AI")
    print("="*70)
    print(f"Epochs: {args.epochs}")
    print(f"Episode Length: {args.episode_length}s")
    print(f"Learning Rate: {args.learning_rate}")
    print("="*70 + "\n")

    # Create environment using direct config import
    from isaaclab_assets.evobot_v1.navigation.velocity.velocity_env_cfg_play import EvobotV1VelocityBalanceEnvCfg

    # Create config
    env_cfg = EvobotV1VelocityBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length

    # Disable all terminations except timeout
    for term_name in dir(env_cfg.terminations):
        if not term_name.startswith('_') and term_name != 'time_out':
            try:
                delattr(env_cfg.terminations, term_name)
            except:
                pass

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print(f"✓ Using environment: EvobotV1VelocityBalanceEnvCfg (Play)")
    print(f"✓ Episode length: {env_cfg.episode_length_s}s")
    print(f"✓ Simulation dt: {env_cfg.sim.dt}s")
    print(f"✓ Disabled terminations (except timeout)")
    print()

    dt = env.step_dt

    # Initialize PID gains (starting point)
    kp = 2.0
    ki = 0.0
    kd = 0.05

    # Best gains tracking
    best_kp, best_ki, best_kd = kp, ki, kd
    best_cost = float('inf')

    # Tracking for anti-stuck mechanism
    no_improvement_count = 0
    last_best_epoch = 0

    # Training history
    history = {
        'epoch': [],
        'kp': [],
        'ki': [],
        'kd': [],
        'cost': [],
        'survived': [],
        'survival_time': [],
        'avg_error': [],
        'max_error': []
    }

    print("Starting auto-tuning...\n")
    print(f"{'Epoch':>5} | {'KP':>7} | {'KI':>7} | {'KD':>7} | {'Cost':>10} | {'Survived':>8} | {'Time':>6} | {'Status':>10}")
    print("-" * 90)

    try:
        for epoch in range(args.epochs):
            # Evaluate current PID
            cost, survived, survival_time, avg_error, max_error = evaluate_pid(
                env, kp, ki, kd, args.max_steps, dt
            )

            # Record history
            history['epoch'].append(epoch)
            history['kp'].append(kp)
            history['ki'].append(ki)
            history['kd'].append(kd)
            history['cost'].append(cost)
            history['survived'].append(survived)
            history['survival_time'].append(survival_time)
            history['avg_error'].append(avg_error)
            history['max_error'].append(max_error)

            # Update best
            status = ""
            if cost < best_cost:
                best_cost = cost
                best_kp, best_ki, best_kd = kp, ki, kd
                last_best_epoch = epoch
                no_improvement_count = 0
                status = "✓ BEST"
            else:
                no_improvement_count += 1

            # Print progress
            survived_str = "YES" if survived else "NO"
            print(f"{epoch:5d} | {kp:7.3f} | {ki:7.3f} | {kd:7.3f} | {cost:10.2f} | {survived_str:>8} | {survival_time:6.2f} | {status:>10}")

            # Auto-adjust gains with anti-stuck mechanism
            if epoch < args.epochs - 1:  # Don't adjust on last epoch

                # Check if stuck (no improvement for 5 consecutive epochs)
                if no_improvement_count >= 5:
                    print(f"  ⚠️  STUCK! No improvement for {no_improvement_count} epochs. RESETTING...")

                    # Strategy 1: Reset to best + large random jump
                    kp = best_kp + np.random.uniform(-2.0, 2.0)
                    ki = best_ki + np.random.uniform(-0.5, 0.5)
                    kd = best_kd + np.random.uniform(-0.3, 0.3)

                    # Ensure valid ranges
                    kp = max(0.1, min(kp, 20.0))
                    ki = max(0.0, min(ki, 5.0))
                    kd = max(0.0, min(kd, 2.0))

                    no_improvement_count = 0  # Reset counter

                elif no_improvement_count >= 3:
                    print(f"  ⚠️  Slow progress. Increasing exploration...")

                    # Strategy 2: Increase exploration (larger learning rate)
                    lr = args.learning_rate * 2.0

                    delta_kp = np.random.uniform(-lr, lr)
                    delta_ki = np.random.uniform(-lr*0.3, lr*0.3)
                    delta_kd = np.random.uniform(-lr*0.3, lr*0.3)

                    kp = max(0.1, min(kp + delta_kp, 20.0))
                    ki = max(0.0, min(ki + delta_ki, 5.0))
                    kd = max(0.0, min(kd + delta_kd, 2.0))

                else:
                    # Normal adjustment with learning rate decay
                    lr = args.learning_rate * (0.9 ** (epoch - last_best_epoch))

                    # Adaptive strategy based on survival
                    if survived:
                        # If survived, fine-tune around current values
                        delta_kp = np.random.uniform(-lr, lr)
                        delta_ki = np.random.uniform(-lr*0.1, lr*0.1)
                        delta_kd = np.random.uniform(-lr*0.1, lr*0.1)
                    else:
                        # If fell, explore more aggressively
                        delta_kp = np.random.uniform(-lr*2, lr*2)
                        delta_ki = np.random.uniform(-lr*0.5, lr*0.5)
                        delta_kd = np.random.uniform(-lr*0.5, lr*0.5)

                    # Update with momentum towards best gains
                    momentum = 0.3  # 30% pull towards best
                    kp = kp + delta_kp + momentum * (best_kp - kp)
                    ki = ki + delta_ki + momentum * (best_ki - ki)
                    kd = kd + delta_kd + momentum * (best_kd - kd)

                    # Clamp to valid ranges
                    kp = max(0.1, min(kp, 20.0))
                    ki = max(0.0, min(ki, 5.0))
                    kd = max(0.0, min(kd, 2.0))

    except KeyboardInterrupt:
        print("\n⚠ Tuning interrupted by user")

    finally:
        # Save results
        print("\n" + "="*70)
        print("AUTO-TUNING COMPLETED")
        print("="*70)
        print(f"Best PID Gains:")
        print(f"  KP = {best_kp:.4f}")
        print(f"  KI = {best_ki:.4f}")
        print(f"  KD = {best_kd:.4f}")
        print(f"  Best Cost = {best_cost:.2f}")
        print("="*70 + "\n")

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "logs", "pid_autotune")
        os.makedirs(output_dir, exist_ok=True)

        # Save CSV
        csv_file = os.path.join(output_dir, f"autotune_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'kp', 'ki', 'kd', 'cost', 'survived',
                           'survival_time', 'avg_error', 'max_error'])
            for i in range(len(history['epoch'])):
                writer.writerow([
                    history['epoch'][i],
                    history['kp'][i],
                    history['ki'][i],
                    history['kd'][i],
                    history['cost'][i],
                    history['survived'][i],
                    history['survival_time'][i],
                    history['avg_error'][i],
                    history['max_error'][i]
                ])
        print(f"✓ Training history saved: {csv_file}")

        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'PID Auto-Tuning Results\nBest: KP={best_kp:.3f}, KI={best_ki:.3f}, KD={best_kd:.3f}',
                     fontsize=14, fontweight='bold')

        # Cost curve
        axes[0, 0].plot(history['epoch'], history['cost'], 'b-', linewidth=2)
        axes[0, 0].axhline(y=best_cost, color='r', linestyle='--', linewidth=2, label=f'Best Cost={best_cost:.1f}')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Cost (Lower is Better)')
        axes[0, 0].set_title('Training Cost')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # PID gains evolution
        axes[0, 1].plot(history['epoch'], history['kp'], 'r-', linewidth=2, label='KP')
        axes[0, 1].plot(history['epoch'], history['ki'], 'g-', linewidth=2, label='KI')
        axes[0, 1].plot(history['epoch'], history['kd'], 'b-', linewidth=2, label='KD')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gain Value')
        axes[0, 1].set_title('PID Gains Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Survival time
        axes[1, 0].plot(history['epoch'], history['survival_time'], 'g-', linewidth=2)
        axes[1, 0].axhline(y=args.episode_length, color='r', linestyle='--',
                          linewidth=2, label=f'Max Time={args.episode_length}s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Survival Time (s)')
        axes[1, 0].set_title('Robot Survival Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Average error
        axes[1, 1].plot(history['epoch'], np.rad2deg(history['avg_error']), 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Average Error (deg)')
        axes[1, 1].set_title('Average Pitch Error')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"autotune_{timestamp}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Training plot saved: {plot_file}")
        plt.close()

        # Test best PID one more time
        print("\n" + "="*70)
        print("TESTING BEST PID GAINS")
        print("="*70)
        print(f"Running final test with KP={best_kp:.4f}, KI={best_ki:.4f}, KD={best_kd:.4f}\n")

        final_cost, final_survived, final_time, final_avg_err, final_max_err = evaluate_pid(
            env, best_kp, best_ki, best_kd, args.max_steps, dt
        )

        print(f"Final Test Results:")
        print(f"  Survived: {'YES' if final_survived else 'NO'}")
        print(f"  Survival Time: {final_time:.2f}s / {args.episode_length}s")
        print(f"  Average Error: {np.rad2deg(final_avg_err):.2f}°")
        print(f"  Max Error: {np.rad2deg(final_max_err):.2f}°")
        print(f"  Cost: {final_cost:.2f}")
        print("="*70 + "\n")

        # Save best PID to config file
        config_file = os.path.join(output_dir, "best_pid_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"# Best PID Configuration\n")
            f.write(f"# Auto-tuned on {timestamp}\n")
            f.write(f"# Training epochs: {args.epochs}\n")
            f.write(f"# Final cost: {best_cost:.2f}\n\n")
            f.write(f"KP = {best_kp:.6f}\n")
            f.write(f"KI = {best_ki:.6f}\n")
            f.write(f"KD = {best_kd:.6f}\n")
        print(f"✓ Best PID config saved: {config_file}\n")

        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
