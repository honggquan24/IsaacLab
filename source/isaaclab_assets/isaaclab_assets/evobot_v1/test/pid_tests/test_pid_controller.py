#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for PID controller with evobot velocity control.

This script runs the environment with PID controller to verify it works correctly.

Usage:
    # Run with default PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_controller.py \
        --num_envs 4

    # Run with custom PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_controller.py \
        --num_envs 4 --kp_linear 3.0 --ki_linear 0.2
"""

import argparse
import torch

# Import Isaac Lab
from isaaclab.app import AppLauncher

# Parse arguments BEFORE launching Isaac Sim
parser = argparse.ArgumentParser(description="Test PID controller for evobot")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--kp_linear", type=float, default=0.0, help="Kp for linear velocity")
parser.add_argument("--ki_linear", type=float, default=0.1, help="Ki for linear velocity")
parser.add_argument("--kd_linear", type=float, default=0.05, help="Kd for linear velocity")
parser.add_argument("--kp_angular", type=float, default=0.1, help="Kp for angular velocity")
parser.add_argument("--ki_angular", type=float, default=0.1, help="Ki for angular velocity")
parser.add_argument("--kd_angular", type=float, default=0.05, help="Kd for angular velocity")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching Isaac Sim
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionManager
from isaaclab.utils import configclass

from isaaclab_assets.evobot_v1.navigation.velocity import EvobotV1VelocityBalanceEnvCfg
from isaaclab_assets.evobot_v1.mdp import VelocityPIDActionTermCfg


def main():
    """Test PID controller."""

    # Create environment config
    env_cfg = EvobotV1VelocityBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda"

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Create PID action config
    pid_cfg = VelocityPIDActionTermCfg(
        asset_name="robot",
        kp_linear=args.kp_linear,
        ki_linear=args.ki_linear,
        kd_linear=args.kd_linear,
        kp_angular=args.kp_angular,
        ki_angular=args.ki_angular,
        kd_angular=args.kd_angular,
        wheel_base=0.2,
        wheel_radius=0.05,
        scale=[300.0, 300.0, 300.0, 100.0, 100.0],
    )

    # Create action manager with PID
    @configclass
    class PIDActionCfg:
        velocity_pid = pid_cfg

    action_manager = ActionManager(PIDActionCfg(), env)

    print("\n" + "="*60)
    print("Testing PID Controller for Evobot Velocity Control")
    print("="*60)
    print(f"Number of environments: {args.num_envs}")
    print(f"PID Gains (Linear):  Kp={args.kp_linear}, Ki={args.ki_linear}, Kd={args.kd_linear}")
    print(f"PID Gains (Angular): Kp={args.kp_angular}, Ki={args.ki_angular}, Kd={args.kd_angular}")
    print("="*60 + "\n")

    # Reset environment
    obs_dict, _ = env.reset()

    # Run simulation
    episode_count = 0
    step_count = 0
    max_episodes = 100

    print("Running simulation... (Ctrl+C to stop)")
    print(f"Target episodes: {max_episodes}\n")

    try:
        while episode_count < max_episodes:
            # Get velocity command from command manager
            vel_cmd = env.command_manager.get_command("base_velocity")

            # Get current robot velocity
            robot = env.scene["robot"]
            vel_current = torch.stack([
                robot.data.root_lin_vel_b[:, 0],  # vx in base frame
                robot.data.root_ang_vel_b[:, 2],  # wz in base frame
            ], dim=-1)

            # Use velocity command as action (PID will handle conversion to wheel velocities)
            action = vel_cmd[:, :2]  # Extract (vx, wz) only

            # Process actions through PID
            action_manager.process_actions(action)
            action_manager.apply_actions()

            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(action)
            dones = terminated | truncated

            # Print info
            if step_count % 50 == 0:
                avg_reward = rewards.mean().item()
                vel_error_lin = torch.abs(vel_cmd[:, 0] - vel_current[:, 0]).mean().item()
                vel_error_ang = torch.abs(vel_cmd[:, 1] - vel_current[:, 1]).mean().item()

                print(f"Step {step_count:4d} | "
                      f"Episodes: {episode_count}/{max_episodes} | "
                      f"Reward: {avg_reward:6.2f} | "
                      f"Lin Error: {vel_error_lin:.3f} m/s | "
                      f"Ang Error: {vel_error_ang:.3f} rad/s")

            # Check for episode completion
            if dones.any():
                done_count = dones.sum().item()
                episode_count += done_count
                print(f"  → {done_count} episode(s) completed")

            step_count += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print(f"\n{'='*60}")
        print(f"Test completed: {episode_count} episodes, {step_count} steps")
        print(f"{'='*60}\n")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
