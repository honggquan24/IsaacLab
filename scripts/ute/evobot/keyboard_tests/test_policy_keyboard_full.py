# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard control for velocity + arm + gripper commands with trained policy.

This script loads a trained policy and allows you to control:
- Base velocity commands (linear x, angular z)
- Arm joint angle (yaw command with PID control)
- Direct gripper action override (both hands)

Usage:
    ./isaaclab.sh -p scripts/ute/evobot/keyboard_tests/test_policy_keyboard_full.py \
        --load_run 2026-01-20_01-01-48 \
        --checkpoint model_1515.pt

Keyboard Controls:
    Base Velocity:
        - Arrow UP      : Move forward
        - Arrow DOWN    : Move backward
        - Arrow LEFT    : Turn left (counter-clockwise)
        - Arrow RIGHT   : Turn right (clockwise)

    Arm Control (Yaw - PID):
        - U : Rotate arm counter-clockwise (increase yaw)
        - O : Rotate arm clockwise (decrease yaw)

    Gripper Action Override (Both hands):
        - Z : MAX OPEN both grippers (+1.0 action)
        - X : MAX CLOSE both grippers (-1.0 action)

    Utility:
        - P : Reset all commands to zero
        - ESC : Exit

Note: Make sure Isaac Sim viewport window has focus!
"""

import argparse
import math
import os

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test trained policy with full keyboard control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-Velocity-Play", help="Task name")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")
parser.add_argument("--vel_sensitivity", type=float, default=0.5, help="Base velocity sensitivity")
parser.add_argument("--arm_sensitivity", type=float, default=0.3, help="Arm rotation sensitivity (rad/step)")
parser.add_argument("--gripper_sensitivity", type=float, default=0.01, help="Gripper height sensitivity (m/step)")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import weakref

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import carb
import omni

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


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
    print("  U                     : Rotate arm counter-clockwise (+ yaw)")
    print("  O                     : Rotate arm clockwise (- yaw)")
    print("\nGripper Action Override (BOTH HANDS):")
    print("  Z                     : MAX OPEN both grippers (+1.0 action)")
    print("  X                     : MAX CLOSE both grippers (-1.0 action)")
    print("\nUtility:")
    print("  P                     : Reset all commands to zero")
    print("  ESC                   : Exit")
    print("\nNote: Make sure Isaac Sim viewport window has focus!")
    print("=" * 80 + "\n")


class FullKeyboardController:
    """Custom keyboard controller for velocity + arm + gripper commands."""

    def __init__(
        self,
        device: str,
        vel_sensitivity: float = 0.1,
        arm_sensitivity: float = 0.1,
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
        self.gripper_action_override = None  # None, +1.0 (max open), or -1.0 (max close)

        # PID controller for arm
        self.arm_kp = 0.1
        self.arm_ki = 0.0
        self.arm_kd = 0.01
        self.arm_integral = 0.0
        self.arm_prev_error = 0.0

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
        if hasattr(self, "_input") and self._input is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        """Reset all commands to zero."""
        self._velocity_cmd.fill(0.0)
        self._arm_yaw_cmd = 0.0
        self.arm_integral = 0.0
        self.arm_prev_error = 0.0

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
        pose[3] = qw  # qw (real part)
        pose[4] = 0.0  # qx
        pose[5] = 0.0  # qy
        pose[6] = qz  # qz (yaw rotation)
        return pose

    def compute_arm_pid_action(self, current_yaw: float, dt: float = 0.01) -> float:
        """Compute PID control for arm yaw.

        Args:
            current_yaw: Current arm yaw angle (radians)
            dt: Time step (seconds)

        Returns:
            Control action for arm joint
        """
        # Compute error
        error = self._arm_yaw_cmd - current_yaw

        # Normalize error to [-pi, pi]
        error = np.arctan2(np.sin(error), np.cos(error))

        # PID terms
        self.arm_integral += error * dt
        derivative = (error - self.arm_prev_error) / dt if dt > 0 else 0.0

        # Compute control output
        output = self.arm_kp * error + self.arm_ki * self.arm_integral + self.arm_kd * derivative

        # Update previous error
        self.arm_prev_error = error

        # Clamp output to [-1, 1]
        output = np.clip(output, -1.0, 1.0)

        return output

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
            "U": ("arm", +self.arm_sensitivity),  # Counter-clockwise
            "O": ("arm", -self.arm_sensitivity),  # Clockwise
            # Gripper action override (both hands simultaneously)
            "Z": ("gripper_both", +1.0),  # Max open (both grippers)
            "X": ("gripper_both", -1.0),  # Max close (both grippers)
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        # Reset all commands
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "P":
                self.reset()
                self.gripper_action_override = None
                print("\n[RESET] All commands reset to zero")
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

        # Handle key release
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._KEY_BINDINGS:
                cmd_type, value = self._KEY_BINDINGS[event.input.name]

                if cmd_type == "velocity":
                    self._velocity_cmd -= value
                elif cmd_type == "gripper_both":
                    # Reset gripper override when key is released
                    self.gripper_action_override = None
                    print("\n[GRIPPER] Released - back to policy control")

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
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
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

    # Main loop
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get keyboard commands
            velocity_cmd = keyboard.get_velocity_command()

            # Override environment commands
            if hasattr(env.unwrapped, "command_manager"):
                # Expand to all environments
                velocity_cmds = velocity_cmd.unsqueeze(0).expand(args_cli.num_envs, -1)
                env.unwrapped.command_manager._terms["velocity_command"].vel_command_b[:, :] = velocity_cmds

            # Get action from policy
            actions = policy(obs)

            # Override arm action with PID controller
            # Get current arm joint position (assume it's index 2 in joint_pos)
            if hasattr(env.unwrapped, "scene"):
                try:
                    robot = env.unwrapped.scene["robot"]
                    current_arm_yaw = robot.data.joint_pos[:, 2]  # Arm yaw joint
                    arm_action = keyboard.compute_arm_pid_action(current_arm_yaw.cpu().numpy(), dt=1 / 60)
                    actions[2] = torch.tensor(arm_action) * 0  # Override arm action (index 2)
                    # actions[2] = 0.0
                except (KeyError, IndexError):
                    pass  # Robot not found or wrong joint index

            # Override gripper actions if Z or NUMPAD_0 is pressed
            if keyboard.gripper_action_override is not None:
                # Assuming gripper actions are the last 2 dimensions [left_gripper, right_gripper]
                # Set both grippers to the override value
                actions[:, -2:] = keyboard.gripper_action_override

            # Step environment

            obs, reward, dones, _ = env_wrapped.step(actions)

            # Print status every 50 steps
            count += 1
            if count % 50 == 0:
                v_x = velocity_cmd[0].item()
                omega_z = velocity_cmd[2].item()
                arm_yaw_target = keyboard._arm_yaw_cmd
                arm_yaw_current = current_arm_yaw[0] if "current_arm_yaw" in locals() else 0.0
                arm_action = actions[0, 2].item() if actions.shape[1] > 2 else 0.0
                grip_override = (
                    keyboard.gripper_action_override if keyboard.gripper_action_override is not None else 0.0
                )

                print(
                    f"\r[CMD] Vel: v_x={v_x:+.2f} ω_z={omega_z:+.2f} | "
                    f"Arm: tgt={arm_yaw_target:+.2f} cur={arm_yaw_current:+.2f} act={arm_action:+.2f} | "
                    f"Grip: {grip_override:+.2f}",
                    end="",
                )

                if torch.is_tensor(reward):
                    mean_reward = reward.mean().item()
                else:
                    mean_reward = float(reward)
                print(f" | Rew: {mean_reward:+.3f}")

            # Handle resets
            if torch.is_tensor(dones) and dones.any():
                print("\n[RESET] Environment terminated. Resetting...")
                obs = env_wrapped.get_observations()

    # Close
    print("\n[INFO] Closing environment...")
    env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
