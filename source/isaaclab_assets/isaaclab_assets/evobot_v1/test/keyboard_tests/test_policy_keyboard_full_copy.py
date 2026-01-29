#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard control for velocity + arm + gripper commands with trained policy.

This script loads a trained policy and allows you to control:
- Base velocity commands (linear x, angular z)
- Arm joint angle (yaw command)
- Direct gripper action override (both hands)
- Action smoothing (exponential moving average)

Usage:
    # Basic usage
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/test_policy_keyboard_full_copy.py \
        --load_run 2026-01-21_08-24-47 \
        --checkpoint model_1410.pt

    # With fixed action smoothing (recommended: 0.3-0.7 for smooth control)
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/test_policy_keyboard_full_copy.py \
        --load_run 2026-01-21_08-24-47 \
        --checkpoint model_1410.pt \
        --action_smoothing 0.5

    # With adaptive smoothing (BEST: auto-adjusts based on command changes)
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/test_policy_keyboard_full_copy.py \
        --load_run 2026-01-21_08-24-47 \
        --checkpoint model_1410.pt \
        --adaptive_smoothing \
        --smoothing_fast 0.2 \
        --smoothing_slow 0.7

Keyboard Controls:
    Base Velocity:
        - Numpad 8 / I      : Move forward
        - Numpad 2 / K      : Move backward
        - Numpad 4 / J      : Turn left (counter-clockwise)
        - Numpad 6 / L      : Turn right (clockwise)

    Arm Control (Yaw):
        - U : Rotate arm counter-clockwise (increase yaw)
        - O : Rotate arm clockwise (decrease yaw)

    Gripper Action Override (Both hands):
        - Z         : MAX OPEN both grippers (+1.0 action)
        - Numpad 0  : MAX CLOSE both grippers (-1.0 action)

    Utility:
        - Numpad 5 / P : Reset all commands to zero
        - ESC          : Exit

Note: Make sure Isaac Sim viewport window has focus!
"""

import argparse
import torch
import os
import numpy as np
import math

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test trained policy with full keyboard control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Velocity-Play", help="Task name")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")
parser.add_argument("--vel_sensitivity", type=float, default=0.5, help="Base velocity sensitivity")
parser.add_argument("--arm_sensitivity", type=float, default=0.3, help="Arm rotation sensitivity (rad/step)")
parser.add_argument("--gripper_sensitivity", type=float, default=0.01, help="Gripper height sensitivity (m/step)")
parser.add_argument("--action_smoothing", type=float, default=0.0, help="Action smoothing factor (0.0=no smoothing, 0.9=heavy smoothing)")
parser.add_argument("--adaptive_smoothing", action="store_true", help="Enable adaptive smoothing (alpha adjusts based on command changes)")
parser.add_argument("--smoothing_fast", type=float, default=0.2, help="Alpha when command changes (fast response)")
parser.add_argument("--smoothing_slow", type=float, default=0.7, help="Alpha when command stable (heavy smoothing)")
parser.add_argument("--command_change_threshold", type=float, default=0.05, help="Threshold to detect command change")
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
import carb
import omni
import weakref

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def print_keyboard_help():
    """Print keyboard control instructions."""
    print("\n" + "=" * 80)
    print("KEYBOARD FULL CONTROL - VELOCITY + ARM (PID) + GRIPPERS")
    print("=" * 80)
    print("Base Velocity Commands:")
    print("  Arrow UP              : Move forward")
    print("  Arrow DOWN            : Move backward")
    print("  Arrow LEFT            : Turn left")
    print("  Arrow RIGHT           : Turn right")
    print("\nArm Control (PID):")
    print("  I                     : Rotate arm counter-clockwise (+ yaw)")
    print("  O                     : Rotate arm clockwise (- yaw)")
    print("\nGripper Action Override (BOTH HANDS):")
    print("  Z                     : MAX OPEN both grippers (+1.0 action)")
    print("  X                     : MAX CLOSE both grippers (-1.0 action)")
    print("\nUtility:")
    print("  P                     : Reset all commands to zero")
    print("  ESC                   : Exit")
    print("\nNote: Make sure Isaac Sim viewport window has focus!")
    print("=" * 80 + "\n")


# ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/test_policy_keyboard_full.py         --load_run 2026-01-20_01-01-48         --checkpoint model_1335.pt

# ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/test/test_policy_keyboard_full_copy.py         --load_run 2026-01-21_08-24-47         --checkpoint model_1410.pt --rendering quality
class FullKeyboardController:
    """Custom keyboard controller for velocity + arm + gripper commands."""

    def __init__(
        self,
        device: str,
        vel_sensitivity: float = 0.1,
        arm_sensitivity: float = 0.01,
        gripper_sensitivity: float = 0.01,
    ):
        """Initialize keyboard controller.

        Args:
            device: Device for tensors (cpu/cuda)
            vel_sensitivity: Base velocity command sensitivity
            arm_sensitivity: Arm rotation sensitivity (radians per step)
            gripper_sensitivity: Gripper height sensitivity (meters per step)
        """
        self.device = device
        self.vel_sensitivity = vel_sensitivity
        self.arm_sensitivity = arm_sensitivity
        self.gripper_sensitivity = gripper_sensitivity

        # Command buffers
        self._velocity_cmd = np.zeros(3)  # [v_x, v_y, omega_z]
        self._arm_yaw_cmd = 0.0  # Target yaw angle for arm

        # Gripper action override (for both hands)
        # Initialize with MAX CLOSE (-1.0) like pressing X key
        self.gripper_action_override = -1.0  # None, +1.0 (max open), or -1.0 (max close)

        # Acquire omniverse keyboard interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Subscribe to keyboard events
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # Additional callbacks
        self._additional_callbacks = {}

        # Create key bindings
        self._create_key_bindings()

    def __del__(self):
        """Release keyboard interface."""
        if hasattr(self, '_input') and self._input is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        """Reset all commands to zero and gripper to MAX CLOSE."""
        self._velocity_cmd.fill(0.0)
        self._arm_yaw_cmd = 0.0
        self._left_gripper_z = 0.0
        self._right_gripper_z = 0.0
        # Reset gripper to MAX CLOSE (like pressing X key)
        self.gripper_action_override = -1.0

    def add_callback(self, key: str, func):
        """Add callback function for specific key."""
        self._additional_callbacks[key] = func

    def get_velocity_command(self) -> torch.Tensor:
        """Get current velocity command [v_x, v_y, omega_z]."""
        return torch.tensor(self._velocity_cmd, dtype=torch.float32, device=self.device)

    def get_arm_command(self) -> torch.Tensor:
        """Get current arm yaw command as pose [x, y, z, qw, qx, qy, qz]."""
        # Return full pose (position + quaternion) with only yaw being meaningful
        # Convert yaw angle to quaternion
        import math
        half_yaw = self._arm_yaw_cmd / 2.0
        qw = math.cos(half_yaw)
        qz = math.sin(half_yaw)

        pose = torch.zeros(7, dtype=torch.float32, device=self.device)
        pose[0:3] = 0.0  # x, y, z position (not used)
        pose[3] = qw     # qw (real part)
        pose[4] = 0.0    # qx
        pose[5] = 0.0    # qy
        pose[6] = qz     # qz (yaw rotation)
        return pose


    def _create_key_bindings(self):
        """Create key bindings for all commands."""
        self._KEY_BINDINGS = {
            # Velocity commands (forward/backward)
            "UP": ("velocity", np.array([1.0, 0.0, 0.0]) * self.vel_sensitivity),
            "DOWN": ("velocity", np.array([-1.0, 0.0, 0.0]) * self.vel_sensitivity),

            # Angular velocity (turn left/right)
            "LEFT": ("velocity", np.array([0.0, 0.0, 1.0]) * self.vel_sensitivity),
            "RIGHT": ("velocity", np.array([0.0, 0.0, -1.0]) * self.vel_sensitivity),

            # Arm rotation (yaw)
            "I": ("arm", +self.arm_sensitivity),  # Counter-clockwise
            "O": ("arm", -self.arm_sensitivity),  # Clockwise

            # Gripper action override (both hands simultaneously)
            "Z": ("gripper_both", +1.0),  # Max open (both grippers)
            "C": ("gripper_both", -1.0),  # Max close (both grippers)
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        # Reset all commands
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "P" or event.input.name == "NUMPAD_5":
                self.reset()
                print("\n[RESET] All commands reset to zero, gripper MAX CLOSE")
                return True

        # Handle key press
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._KEY_BINDINGS:
                cmd_type, value = self._KEY_BINDINGS[event.input.name]

                if cmd_type == "velocity":
                    self._velocity_cmd += value
                elif cmd_type == "arm":
                    self._arm_yaw_cmd += value
                    # Clamp to ±π
                    self._arm_yaw_cmd = np.clip(self._arm_yaw_cmd, -math.pi, math.pi)
                elif cmd_type == "gripper_both":
                    # Set gripper action override for both hands
                    self.gripper_action_override = value
                    if value > 0:
                        print(f"\n[GRIPPER] Both hands MAX OPEN (+{value:.1f})")
                    else:
                        print(f"\n[GRIPPER] Both hands MAX CLOSE ({value:.1f})")

        # Handle key release (only for velocity - release stops motion)
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._KEY_BINDINGS:
                cmd_type, value = self._KEY_BINDINGS[event.input.name]

                if cmd_type == "velocity":
                    self._velocity_cmd -= value

        # Additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        return True


def load_policy(env, agent_cfg, checkpoint_path: str):
    """Load trained policy from checkpoint."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return None

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
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
    print("EVOBOT V1 VELOCITY - FULL KEYBOARD CONTROL")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 80)

    # Load policy
    if args_cli.load_run is None:
        print("\n[ERROR] No checkpoint specified!")
        print("[ERROR] Usage: --load_run <run_name> --checkpoint model_500.pt\n")
        env.close()
        return

    # Get agent configuration
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    try:
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    except Exception as e:
        print(f"[ERROR] Failed to load agent config: {e}")
        env.close()
        return

    # Wrap environment for RSL-RL
    env_wrapped = RslRlVecEnvWrapper(env)

    # Construct checkpoint path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    checkpoint_file = args_cli.checkpoint if args_cli.checkpoint else "model_.*\\.pt"
    checkpoint_path = get_checkpoint_path(log_root_path, args_cli.load_run, checkpoint_file)

    # Load policy
    policy = load_policy(env_wrapped, agent_cfg, checkpoint_path)
    if policy is None:
        print("[ERROR] Failed to load policy. Exiting.")
        env.close()
        return

    # Initialize keyboard controller
    keyboard = FullKeyboardController(
        device=args_cli.device,
        vel_sensitivity=args_cli.vel_sensitivity,
        arm_sensitivity=args_cli.arm_sensitivity,
        gripper_sensitivity=args_cli.gripper_sensitivity,
    )

    # Add ESC callback
    def exit_callback():
        print("\n[ESC] Exiting...")
        simulation_app.close()
    keyboard.add_callback("ESCAPE", exit_callback)

    # Print help
    print_keyboard_help()

    # Reset environment
    env.reset()
    obs = env_wrapped.get_observations()
    print("[INFO] Environment ready. Use keyboard to control...\n")

    # Action smoothing buffer
    alpha = args_cli.action_smoothing
    prev_actions = None
    prev_velocity_cmd = None
    use_adaptive = args_cli.adaptive_smoothing

    if alpha > 0.0 or use_adaptive:
        if use_adaptive:
            print(f"[INFO] Adaptive EMA smoothing enabled:")
            print(f"      - Fast alpha (command change): {args_cli.smoothing_fast:.2f}")
            print(f"      - Slow alpha (stable): {args_cli.smoothing_slow:.2f}")
            print(f"      - Change threshold: {args_cli.command_change_threshold:.3f}")
        else:
            print(f"[INFO] Fixed EMA smoothing enabled: alpha={alpha:.2f}")
        print(f"      Formula: action_smooth = alpha * prev_action + (1-alpha) * current_action\n")

    # Main loop
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get keyboard commands
            velocity_cmd = keyboard.get_velocity_command()
            arm_cmd = keyboard.get_arm_command()

            # Override environment commands
            if hasattr(env.unwrapped, "command_manager"):
                # Expand to all environments
                velocity_cmds = velocity_cmd.unsqueeze(0).expand(args_cli.num_envs, -1)
                arm_cmds = arm_cmd.unsqueeze(0).expand(args_cli.num_envs, -1)

                # Set commands in command manager
                env.unwrapped.command_manager._terms["velocity_command"].vel_command_b[:, :] = velocity_cmds

                # Arm pose command (full pose: x, y, z, qw, qx, qy, qz in base frame)
                env.unwrapped.command_manager._terms["arm_ee_pose"].pose_command_b[:, :] = arm_cmds

            # Get action from policy
            actions = policy(obs)

            # Override gripper actions if Z or NUMPAD_0 is pressed
            if keyboard.gripper_action_override is not None:
                # Assuming gripper actions are the last 2 dimensions [left_gripper, right_gripper]
                # Set both grippers to the override value
                actions[:, -2:] = keyboard.gripper_action_override

            # Apply action smoothing (exponential moving average)
            if alpha > 0.0 or use_adaptive:
                if prev_actions is None:
                    # First step: initialize with current action
                    prev_actions = actions.clone()
                    if use_adaptive:
                        prev_velocity_cmd = velocity_cmd.clone()
                else:
                    # Determine alpha (fixed or adaptive)
                    current_alpha = alpha

                    if use_adaptive:
                        # Adaptive smoothing: adjust alpha based on command changes
                        if prev_velocity_cmd is not None:
                            # Detect velocity command change
                            cmd_diff = torch.abs(velocity_cmd - prev_velocity_cmd).max().item()

                            if cmd_diff > args_cli.command_change_threshold:
                                # Command changed: use fast alpha (low value = fast response)
                                current_alpha = args_cli.smoothing_fast
                            else:
                                # Command stable: use slow alpha (high value = heavy smoothing)
                                current_alpha = args_cli.smoothing_slow

                            prev_velocity_cmd = velocity_cmd.clone()
                        else:
                            current_alpha = args_cli.smoothing_slow

                    # Apply smoothing: action_smooth = alpha * prev_action + (1-alpha) * current_action
                    actions = current_alpha * prev_actions + (1.0 - current_alpha) * actions
                    prev_actions = actions.clone()

            # Step environment
            obs, reward, dones, _ = env_wrapped.step(actions)

            # Print status every 50 steps
            count += 1
            if count % 50 == 0:
                v_x = velocity_cmd[0].item()
                omega_z = velocity_cmd[2].item()
                arm_yaw = arm_cmd[6].item()
                grip_override = keyboard.gripper_action_override if keyboard.gripper_action_override is not None else 0.0

                print(f"\r[CMD] Vel: v_x={v_x:+.2f} ω_z={omega_z:+.2f} | Arm: yaw={arm_yaw:+.2f} | "
                      f"Grip: {grip_override:+.2f}", end="")

                if torch.is_tensor(reward):
                    mean_reward = reward.mean().item()
                else:
                    mean_reward = float(reward)
                print(f" | Rew: {mean_reward:+.3f}")

            # Handle resets
            if torch.is_tensor(dones) and dones.any():
                print("\n[RESET] Environment terminated. Resetting...")
                # Reset keyboard commands (like pressing P)
                keyboard.reset()
                print("[RESET] All commands reset to zero, gripper MAX CLOSE")
                # Reset action smoothing buffer
                prev_actions = None
                obs = env_wrapped.get_observations()

    # Close
    print("\n[INFO] Closing environment...")
    env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
