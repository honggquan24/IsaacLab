# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to control robot with multiple control modes."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Multi-mode manual control for Isaac Lab environments.")
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
import numpy as np
import torch
import weakref
from enum import Enum

import carb

import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


class ControlMode(Enum):
    """Control modes for the robot."""

    VELOCITY = 1  # Direct velocity control
    POSITION = 2  # Position control
    ZERO = 3  # Send zero actions


class MultiModeController:
    """Multi-mode keyboard controller."""

    def __init__(self, action_dim: int, device: str):
        """Initialize multi-mode controller.

        Args:
            action_dim: Dimension of action space
            device: Device to create tensors on
        """
        self.action_dim = action_dim
        self.device = device

        # Current control mode
        self.mode = ControlMode.VELOCITY

        # Initialize keyboard device
        keyboard_cfg = Se2KeyboardCfg(
            v_x_sensitivity=0.8,
            v_y_sensitivity=0.4,
            omega_z_sensitivity=1.0,
            sim_device=device,
        )
        self.keyboard = Se2Keyboard(cfg=keyboard_cfg)

        # Add mode switching callbacks
        self._setup_mode_callbacks()

        # Position target (for position control mode)
        self.target_position = np.array([0.0, 0.0, 0.0])  # [x, y, heading]
        self.position_increment = 0.2

        print("\n" + "=" * 80)
        print("MULTI-MODE CONTROL")
        print("=" * 80)
        print(str(self.keyboard))
        print("\nMode Switching:")
        print("  1: VELOCITY mode - Direct velocity control")
        print("  2: POSITION mode - Target position control")
        print("  3: ZERO mode - Send zero actions")
        print("\nPosition Mode Controls (when in mode 2):")
        print("  I/K: Increase/Decrease target X")
        print("  J/L: Increase/Decrease target Y")
        print("  U/O: Increase/Decrease target heading")
        print("\nOther:")
        print("  P: Print current status")
        print("  ESC: Exit")
        print("=" * 80 + "\n")

    def _setup_mode_callbacks(self):
        """Setup keyboard callbacks for mode switching."""
        self.keyboard.add_callback("1", lambda: self._switch_mode(ControlMode.VELOCITY))
        self.keyboard.add_callback("2", lambda: self._switch_mode(ControlMode.POSITION))
        self.keyboard.add_callback("3", lambda: self._switch_mode(ControlMode.ZERO))
        self.keyboard.add_callback("P", self._print_status)
        self.keyboard.add_callback("I", lambda: self._adjust_target(0, self.position_increment))
        self.keyboard.add_callback("K", lambda: self._adjust_target(0, -self.position_increment))
        self.keyboard.add_callback("J", lambda: self._adjust_target(1, self.position_increment))
        self.keyboard.add_callback("L", lambda: self._adjust_target(1, -self.position_increment))
        self.keyboard.add_callback("U", lambda: self._adjust_target(2, 0.1))
        self.keyboard.add_callback("O", lambda: self._adjust_target(2, -0.1))

    def _switch_mode(self, new_mode: ControlMode):
        """Switch control mode."""
        self.mode = new_mode
        print(f"\n[MODE CHANGE] Switched to {self.mode.name} mode")

    def _adjust_target(self, idx: int, delta: float):
        """Adjust target position."""
        if self.mode == ControlMode.POSITION:
            self.target_position[idx] += delta
            print(f"\n[TARGET] Position: [{self.target_position[0]:.2f}, {self.target_position[1]:.2f}], "
                  f"Heading: {self.target_position[2]:.2f}")

    def _print_status(self):
        """Print current controller status."""
        cmd = self.keyboard.advance().cpu().numpy()
        print(f"\n[STATUS]")
        print(f"  Mode: {self.mode.name}")
        print(f"  Keyboard command: v_x={cmd[0]:.2f}, v_y={cmd[1]:.2f}, omega_z={cmd[2]:.2f}")
        if self.mode == ControlMode.POSITION:
            print(f"  Target position: x={self.target_position[0]:.2f}, y={self.target_position[1]:.2f}, "
                  f"heading={self.target_position[2]:.2f}")

    def get_actions(self, current_pos=None) -> torch.Tensor:
        """Get actions based on current mode.

        Args:
            current_pos: Current position [x, y, heading] for position control mode

        Returns:
            Action tensor for the environment
        """
        if self.mode == ControlMode.VELOCITY:
            # Direct velocity control
            keyboard_command = self.keyboard.advance()
            if self.action_dim == 3:
                actions = keyboard_command.unsqueeze(0)
            else:
                # Try to map to first 3 dimensions
                actions = torch.zeros(1, self.action_dim, device=self.device)
                actions[:, :3] = keyboard_command.unsqueeze(0)

        elif self.mode == ControlMode.POSITION:
            # Position control - compute velocity towards target
            if current_pos is not None:
                # Simple proportional controller
                error = self.target_position - current_pos
                # Clamp to max velocities
                vel_x = np.clip(error[0] * 2.0, -0.8, 0.8)
                vel_y = np.clip(error[1] * 2.0, -0.4, 0.4)
                omega_z = np.clip(error[2] * 2.0, -1.0, 1.0)

                command = torch.tensor([vel_x, vel_y, omega_z], device=self.device)
            else:
                # No position feedback, use zero
                command = torch.zeros(3, device=self.device)

            if self.action_dim == 3:
                actions = command.unsqueeze(0)
            else:
                actions = torch.zeros(1, self.action_dim, device=self.device)
                actions[:, :3] = command.unsqueeze(0)

        else:  # ZERO mode
            actions = torch.zeros(1, self.action_dim, device=self.device)

        return actions


def main():
    """Multi-mode control agent with Isaac Lab environment."""
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
    print("=" * 80)

    # Initialize controller
    action_dim = env.action_space.shape[-1]
    controller = MultiModeController(action_dim=action_dim, device=env.unwrapped.device)

    # reset environment
    obs, info = env.reset()

    # simulate environment
    count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Get robot position if available (for position control mode)
            # This assumes the observation contains position information
            current_pos = None
            # TODO: Extract position from obs if needed for position control

            # Get actions based on mode
            single_action = controller.get_actions(current_pos)

            # Repeat for all environments
            actions = single_action.repeat(args_cli.num_envs, 1)

            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)

            # print status every 200 steps
            count += 1
            if count % 200 == 0:
                action_values = actions[0].cpu().numpy()
                print(
                    f"Step {count:6d} | Mode: {controller.mode.name:8s} | "
                    f"Actions: [{action_values[0]:+.2f}, {action_values[1]:+.2f}, {action_values[2]:+.2f}] | "
                    f"Reward: {reward[0].item():+.2f}"
                )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
