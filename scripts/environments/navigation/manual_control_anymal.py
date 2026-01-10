# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to manually control ANYmal-C navigation using keyboard with trained policy."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Manual control of ANYmal-C navigation with trained policy.")
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

from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# Import RL library (RSL-RL)
from rsl_rl.runners import OnPolicyRunner


def main():
    """Manual control of ANYmal-C navigation with trained policy."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Modify configuration for manual control
    env_cfg.terminations.time_out = None  # No timeout
    # Set very large resampling time so we control the target manually
    env_cfg.commands.pose_command.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.pose_command.debug_vis = True  # Enable visualization

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Load trained policy
    if args_cli.load_run:
        # Get checkpoint path
        log_root_path = os.path.join("logs", "rsl_rl", args_cli.task)
        log_dir = os.path.join(log_root_path, args_cli.load_run)

        # Check if checkpoint exists
        checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)
        if not os.path.exists(checkpoint_path):
            # Try to find the latest checkpoint
            import glob
            checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"[INFO] Using latest checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"No checkpoint found in {log_dir}")

        # Load configuration
        resume_path = os.path.join(log_dir, "params", "env.pkl")
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

        # Create policy runner
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)

        # Load policy weights
        print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
        ppo_runner.load(checkpoint_path)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    else:
        print("[WARNING] No checkpoint provided. Using random actions.")
        policy = None

    # Create keyboard interface for target control
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.5,  # Adjust sensitivity as needed
        v_y_sensitivity=0.5,
        omega_z_sensitivity=0.5
    )
    keyboard = Se2Keyboard(keyboard_cfg)

    # Initialize target position (relative to current robot position)
    target_delta = np.zeros(3)  # [dx, dy, dyaw]

    # Callback for resetting target
    def reset_target():
        nonlocal target_delta
        target_delta = np.zeros(3)
        print("[INFO] Target reset to robot position")

    keyboard.add_callback("R", reset_target)

    # Reset environment
    obs, _ = env.reset()
    keyboard.reset()

    print("\n" + "="*80)
    print("Manual Control of ANYmal-C Navigation")
    print("="*80)
    print("Controls:")
    print("  Arrow Keys / Numpad 8,2,4,6: Move target position (x, y)")
    print("  Z / X (or Numpad 7/9):        Rotate target heading")
    print("  R:                            Reset target to robot position")
    print("  L:                            Stop target movement")
    print("  ESC:                          Quit")
    print("="*80 + "\n")

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get keyboard command (velocity command for target)
            target_vel = keyboard.advance().cpu().numpy()

            # Update target position incrementally
            dt = env_cfg.sim.dt * env_cfg.decimation
            target_delta += target_vel * dt

            # Get robot base position
            robot = env.scene["robot"]
            robot_pos = robot.data.root_pos_w[0, :2].cpu().numpy()  # [x, y]
            robot_heading = robot.data.heading_w[0].cpu().item()

            # Calculate absolute target position
            target_pos = robot_pos + target_delta[:2]
            target_heading = robot_heading + target_delta[2]

            # Manually set the command in the environment
            env.command_manager._terms["pose_command"]._pos_command_w[:, 0] = target_pos[0]
            env.command_manager._terms["pose_command"]._pos_command_w[:, 1] = target_pos[1]
            env.command_manager._terms["pose_command"]._heading_command_w[:] = target_heading

            # Get policy action
            if policy:
                # Prepare observations
                if isinstance(obs, dict):
                    obs_tensor = obs["policy"]
                else:
                    obs_tensor = obs

                # Get action from policy
                actions = policy(obs_tensor)
            else:
                # Random actions if no policy loaded
                actions = 2.0 * torch.rand(env.num_envs, env.action_space.shape[0], device=env.device) - 1.0

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Print info every 100 steps
            if env.episode_length_buf[0] % 100 == 0:
                distance = np.linalg.norm(robot_pos - target_pos)
                print(f"Step {env.episode_length_buf[0].item():5d} | "
                      f"Target: ({target_pos[0]:+.2f}, {target_pos[1]:+.2f}, {target_heading:+.2f}) | "
                      f"Distance: {distance:.2f}m | "
                      f"Reward: {rewards[0].item():+.2f}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation app
    simulation_app.close()