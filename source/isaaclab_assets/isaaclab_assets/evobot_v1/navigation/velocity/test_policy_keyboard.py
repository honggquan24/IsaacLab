#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard control for velocity commands with trained policy.

This script loads a trained policy and allows you to control the velocity commands
using keyboard input. The policy executes actions to track the commanded velocities.

Usage:
    # ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_policy_keyboard.py \
    #     --load_run <run_name> \
    #     --checkpoint model_500.pt

Keyboard Controls:
    - Arrow Up / Numpad 8: Increase forward velocity
    - Arrow Down / Numpad 2: Decrease forward velocity
    - Z / Numpad 7: Increase left turn rate (counter-clockwise)
    - X / Numpad 9: Increase right turn rate (clockwise)
    - L: Reset all velocities to zero
    - ESC: Exit

Note: These key bindings are chosen to avoid conflicts with Isaac Sim UI shortcuts.
"""

import argparse
import torch
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test trained policy with keyboard velocity commands")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Velocity", help="Task name")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name (e.g., 2025-01-15_10-30-45)")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")
parser.add_argument("--sensitivity", type=float, default=0.5, help="Command sensitivity")
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
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def print_keyboard_help():
    """Print keyboard control instructions."""
    print("\n" + "=" * 80)
    print("KEYBOARD VELOCITY COMMAND CONTROL")
    print("=" * 80)
    print("Velocity Commands:")
    print("  Arrow Up / Numpad 8   : Move forward")
    print("  Arrow Down / Numpad 2 : Move backward")
    print("  Z / Numpad 7          : Turn left (counter-clockwise)")
    print("  X / Numpad 9          : Turn right (clockwise)")
    print("  L                     : Reset all velocities to zero")
    print("\nUtility:")
    print("  ESC                   : Exit")
    print("\nNote: Make sure Isaac Sim viewport window has focus!")
    print("=" * 80 + "\n")


def load_policy(env, agent_cfg, checkpoint_path: str):
    """Load trained policy from checkpoint using RSL-RL runner.

    Args:
        env: Wrapped environment instance (RslRlVecEnvWrapper)
        agent_cfg: Agent configuration
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded policy or None if checkpoint doesn't exist
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print("[WARNING] Cannot load policy without valid checkpoint")
        return None

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # Create runner and load checkpoint (same as play.py)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print("[INFO] Policy loaded successfully")
    return policy


def main():
    """Main function."""

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print environment info
    print("\n" + "=" * 80)
    print("EVOBOT V1 VELOCITY - POLICY WITH KEYBOARD COMMANDS")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 80)

    # Load agent config
    agent_cfg = None
    policy = None

    if args_cli.load_run is not None:
        # Get agent configuration from registry
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        try:
            agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        except Exception as e:
            print(f"[ERROR] Failed to load agent config: {e}")
            print("[ERROR] Cannot load policy without agent config")
            env.close()
            return

        # Wrap environment for RSL-RL (required by OnPolicyRunner)
        env_wrapped = RslRlVecEnvWrapper(env)

        # Construct checkpoint path
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

        # Build full checkpoint path
        checkpoint_file = args_cli.checkpoint if args_cli.checkpoint else "model_.*\\.pt"
        checkpoint_path = get_checkpoint_path(log_root_path, args_cli.load_run, checkpoint_file)

        # Load policy
        policy = load_policy(env_wrapped, agent_cfg, checkpoint_path)

        if policy is None:
            print("[ERROR] Failed to load policy. Exiting.")
            env.close()
            return
    else:
        print("\n[ERROR] No checkpoint specified!")
        print("[ERROR] This script requires a trained policy.")
        print("[INFO] Usage: --load_run <run_name> --checkpoint model_500.pt\n")
        env.close()
        return

    # Initialize keyboard controller for velocity commands
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=1.0 * args_cli.sensitivity,
        v_y_sensitivity=0.0,  # Disabled (differential drive robot)
        omega_z_sensitivity=1.0 * args_cli.sensitivity,
        sim_device=args_cli.device,
    )
    keyboard = Se2Keyboard(cfg=keyboard_cfg)

    # Add ESC callback to exit
    def exit_callback():
        print("\n[ESC] Exiting...")
        simulation_app.close()
    keyboard.add_callback("ESCAPE", exit_callback)

    # Print keyboard help
    print_keyboard_help()

    # Reset environment
    env.reset()
    obs = env_wrapped.get_observations()
    print("[INFO] Environment ready. Control velocity commands with keyboard...\n")

    # Main loop
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get keyboard velocity commands [v_x, v_y, omega_z]
            keyboard_cmd = keyboard.advance()

            # Override environment's velocity commands
            # Se2Keyboard returns tensor on correct device already
            if hasattr(env.unwrapped, "command_manager"):
                # Directly set velocity commands in command manager
                # Expand to all environments
                velocity_commands = keyboard_cmd.unsqueeze(0).expand(args_cli.num_envs, -1)
                env.unwrapped.command_manager._terms["base_velocity"].vel_command_b[:, :] = velocity_commands
            else:
                print("[WARNING] Environment does not have command_manager")

            # Get action from trained policy (wrapped env returns properly formatted obs)
            actions = policy(obs)

            # Step environment
            obs, reward, dones, _ = env_wrapped.step(actions)

            # Print status every 50 steps
            count += 1
            if count % 50 == 0:
                # Print current velocity command
                v_x = keyboard_cmd[0].item()
                omega_z = keyboard_cmd[2].item()
                print(f"\r[CMD] v_x: {v_x:+.2f} m/s | ω_z: {omega_z:+.2f} rad/s", end="")

                if torch.is_tensor(reward):
                    mean_reward = reward.mean().item()
                else:
                    mean_reward = float(reward)
                print(f" | Reward: {mean_reward:+.3f}")

            # Handle resets (wrapped env uses 'dones' instead of terminated/truncated)
            if torch.is_tensor(dones) and dones.any():
                print("\n[RESET] Environment terminated/truncated. Resetting...")
                # Wrapped env handles partial resets automatically
                # Just get fresh observations
                obs = env_wrapped.get_observations()

    # Close environment
    print("\n[INFO] Closing environment...")
    env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    # Run main
    main()
    # Close sim app
    simulation_app.close()
