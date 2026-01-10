# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to interactively test ANYmal-C navigation by clicking target positions."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive navigation testing with mouse click targets.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Flat-Anymal-C-v0", help="Task name.")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load.")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="Checkpoint file to load.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# Import RL library (RSL-RL)
from rsl_rl.runners import OnPolicyRunner


def main():
    """Interactive navigation testing."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Modify configuration
    env_cfg.terminations.time_out = None
    env_cfg.commands.pose_command.debug_vis = True

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Load trained policy
    policy = None
    if args_cli.load_run:
        log_root_path = os.path.join("logs", "rsl_rl", args_cli.task)
        log_dir = os.path.join(log_root_path, args_cli.load_run)

        checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)
        if not os.path.exists(checkpoint_path):
            import glob
            checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"[INFO] Using latest checkpoint: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
            ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
            print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
            ppo_runner.load(checkpoint_path)
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Reset environment
    obs, _ = env.reset()

    # List of waypoints to visit (in world coordinates)
    waypoints = [
        [2.0, 0.0, 0.0],    # Forward 2m
        [2.0, 2.0, 1.57],   # Turn right, go 2m
        [0.0, 2.0, 3.14],   # Turn around, go back
        [0.0, 0.0, 0.0],    # Return to start
    ]
    current_waypoint_idx = 0

    print("\n" + "="*80)
    print("Interactive ANYmal-C Navigation")
    print("="*80)
    print("The robot will navigate through predefined waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i}: x={wp[0]:+.1f}m, y={wp[1]:+.1f}m, heading={wp[2]:+.2f}rad")
    print("\nPress 'N' to skip to next waypoint")
    print("="*80 + "\n")

    step_count = 0
    reached_threshold = 0.3  # meters

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get current waypoint
            target = waypoints[current_waypoint_idx]

            # Set command
            env.command_manager._terms["pose_command"]._pos_command_w[:, 0] = target[0]
            env.command_manager._terms["pose_command"]._pos_command_w[:, 1] = target[1]
            env.command_manager._terms["pose_command"]._heading_command_w[:] = target[2]

            # Get policy action
            if policy:
                if isinstance(obs, dict):
                    obs_tensor = obs["policy"]
                else:
                    obs_tensor = obs
                actions = policy(obs_tensor)
            else:
                actions = 2.0 * torch.rand(env.num_envs, env.action_space.shape[0], device=env.device) - 1.0

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1

            # Check if reached waypoint
            robot = env.scene["robot"]
            robot_pos = robot.data.root_pos_w[0, :2].cpu().numpy()
            distance = np.linalg.norm(robot_pos - np.array(target[:2]))

            if distance < reached_threshold:
                print(f"[SUCCESS] Reached waypoint {current_waypoint_idx}!")
                current_waypoint_idx = (current_waypoint_idx + 1) % len(waypoints)
                step_count = 0

            # Print info
            if step_count % 50 == 0:
                print(f"Target WP{current_waypoint_idx}: ({target[0]:+.1f}, {target[1]:+.1f}) | "
                      f"Robot: ({robot_pos[0]:+.2f}, {robot_pos[1]:+.2f}) | "
                      f"Dist: {distance:.2f}m | Reward: {rewards[0].item():+.2f}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
