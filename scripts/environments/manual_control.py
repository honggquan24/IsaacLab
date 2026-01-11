# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to manually control a robot with keyboard input."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Manual control for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity multiplier for keyboard commands.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Manual control agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print("\n" + "=" * 80)
    print("ENVIRONMENT INFORMATION")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    print("=" * 80)

    # Initialize keyboard controller
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.8 * args_cli.sensitivity,
        v_y_sensitivity=0.4 * args_cli.sensitivity,
        omega_z_sensitivity=1.0 * args_cli.sensitivity,
        sim_device=args_cli.device,
    )
    keyboard = Se2Keyboard(cfg=keyboard_cfg)
    print("\n" + str(keyboard))
    print("\n" + "=" * 80)
    print("MANUAL CONTROL ACTIVE")
    print("=" * 80)
    print("Use keyboard to control the robot.")
    print("Press ESC to exit.")
    print("=" * 80 + "\n")

    # Determine how to map keyboard commands to action space
    action_dim = env.action_space.shape[-1]
    print(f"[INFO] Action dimension: {action_dim}")

    # Check if this is a navigation task or locomotion task
    # Navigation tasks typically have smaller action spaces (e.g., 3 for vel_x, vel_y, yaw_rate)
    # Locomotion tasks have larger action spaces (e.g., 12 for ANYmal-C joint commands)

    # reset environment
    obs, info = env.reset()

    # simulate environment
    count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command [v_x, v_y, omega_z]
            keyboard_command = keyboard.advance()

            # Map keyboard command to action space
            if action_dim == 3:
                # Direct velocity command (navigation)
                actions = keyboard_command.unsqueeze(0).repeat(args_cli.num_envs, 1)
            elif action_dim == 12:
                # Joint-level control (ANYmal-C has 12 joints)
                # For locomotion, we need a policy to convert vel commands to joint commands
                # For now, send zero actions as placeholder
                actions = torch.zeros(args_cli.num_envs, action_dim, device=env.unwrapped.device)
                print(
                    "[WARNING] This environment uses joint-level control. "
                    "Manual control is not directly supported. "
                    "Please use a trained policy or implement a controller."
                )
            else:
                # Unknown action space, try to use first 3 dimensions
                actions = torch.zeros(args_cli.num_envs, action_dim, device=env.unwrapped.device)
                actions[:, :3] = keyboard_command.unsqueeze(0).repeat(args_cli.num_envs, 1)

            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)

            # print status every 100 steps
            count += 1
            if count % 100 == 0:
                cmd = keyboard_command.cpu().numpy()
                print(
                    f"Step {count:6d} | Command: "
                    f"v_x={cmd[0]:+.2f} v_y={cmd[1]:+.2f} omega_z={cmd[2]:+.2f} | "
                    f"Reward: {reward[0].item():.2f}"
                )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
