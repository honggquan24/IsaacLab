# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive play script for ANYmal-C navigation.
Allows setting target positions via console input or keyboard.
"""

import argparse
import torch
import threading

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive play for ANYmal-C navigation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Flat-Anymal-C-v0", help="Task name.")
parser.add_argument("--load_run", type=str, required=True, help="Name of the run folder to load.")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="Checkpoint file to load.")
parser.add_argument("--mode", type=str, default="keyboard", choices=["keyboard", "console", "waypoint"],
                    help="Control mode: keyboard (arrow keys), console (text input), waypoint (predefined)")
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from rsl_rl.runners import OnPolicyRunner


class NavigationController:
    """Controller for interactive navigation."""

    def __init__(self, env, mode="keyboard"):
        self.env = env
        self.mode = mode
        self.target_pos = np.array([0.0, 0.0])
        self.target_heading = 0.0
        self.keyboard = None
        self.console_input = None

        if mode == "keyboard":
            self._setup_keyboard()
        elif mode == "console":
            self._setup_console()
        elif mode == "waypoint":
            self._setup_waypoints()

    def _setup_keyboard(self):
        """Setup keyboard control."""
        keyboard_cfg = Se2KeyboardCfg(
            v_x_sensitivity=0.3,
            v_y_sensitivity=0.3,
            omega_z_sensitivity=0.5
        )
        self.keyboard = Se2Keyboard(keyboard_cfg)

        def reset_target():
            self.target_pos = np.zeros(2)
            self.target_heading = 0.0
            print("[INFO] Target reset")

        self.keyboard.add_callback("R", reset_target)
        self.keyboard.reset()

        print("\n[Keyboard Mode]")
        print("  Arrow Keys: Move target")
        print("  Z/X: Rotate target")
        print("  R: Reset target to origin")
        print("  L: Stop movement\n")

    def _setup_console(self):
        """Setup console input in separate thread."""
        self.console_input = {"x": 0.0, "y": 0.0, "heading": 0.0}

        def input_thread():
            print("\n[Console Mode]")
            print("Enter target coordinates (x y heading_deg):")
            print("Example: 2.0 1.5 45")
            while True:
                try:
                    inp = input("> ").strip().split()
                    if len(inp) >= 2:
                        self.console_input["x"] = float(inp[0])
                        self.console_input["y"] = float(inp[1])
                        self.console_input["heading"] = np.deg2rad(float(inp[2])) if len(inp) > 2 else 0.0
                        print(f"[INFO] Target set to: ({self.console_input['x']:.2f}, "
                              f"{self.console_input['y']:.2f}, {np.rad2deg(self.console_input['heading']):.1f}°)")
                except (ValueError, EOFError, KeyboardInterrupt):
                    break

        thread = threading.Thread(target=input_thread, daemon=True)
        thread.start()

    def _setup_waypoints(self):
        """Setup predefined waypoints."""
        self.waypoints = [
            {"pos": [2.0, 0.0], "heading": 0.0, "name": "Forward"},
            {"pos": [2.0, 2.0], "heading": np.pi/2, "name": "Right"},
            {"pos": [0.0, 2.0], "heading": np.pi, "name": "Back"},
            {"pos": [0.0, 0.0], "heading": -np.pi/2, "name": "Origin"},
        ]
        self.current_waypoint_idx = 0
        self.reached_threshold = 0.5

        print("\n[Waypoint Mode]")
        print("Predefined waypoints:")
        for i, wp in enumerate(self.waypoints):
            print(f"  {i}. {wp['name']}: ({wp['pos'][0]:.1f}, {wp['pos'][1]:.1f}), {np.rad2deg(wp['heading']):.0f}°")
        print()

    def get_target(self) -> tuple:
        """Get current target position and heading."""
        if self.mode == "keyboard":
            # Incremental control
            dt = self.env.cfg.sim.dt * self.env.cfg.decimation
            vel_cmd = self.keyboard.advance().cpu().numpy()
            self.target_pos += vel_cmd[:2] * dt
            self.target_heading += vel_cmd[2] * dt
            return self.target_pos[0], self.target_pos[1], self.target_heading

        elif self.mode == "console":
            # Absolute target from console input
            return self.console_input["x"], self.console_input["y"], self.console_input["heading"]

        elif self.mode == "waypoint":
            # Waypoint navigation
            wp = self.waypoints[self.current_waypoint_idx]
            target_pos = np.array(wp["pos"])
            target_heading = wp["heading"]

            # Check if reached
            robot = self.env.scene["robot"]
            robot_pos = robot.data.root_pos_w[0, :2].cpu().numpy()
            distance = np.linalg.norm(robot_pos - target_pos)

            if distance < self.reached_threshold:
                print(f"[SUCCESS] Reached waypoint: {wp['name']}")
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)

            return target_pos[0], target_pos[1], target_heading

        return 0.0, 0.0, 0.0


def main():
    """Main function."""

    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None
    env_cfg.commands.pose_command.debug_vis = True
    env_cfg.commands.pose_command.resampling_time_range = (1.0e9, 1.0e9)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Load trained policy
    log_root_path = os.path.join("logs", "rsl_rl", args_cli.task)
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)

    if not os.path.exists(checkpoint_path):
        import glob
        checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {log_dir}")
        checkpoint_path = max(checkpoints, key=os.path.getctime)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Create controller
    controller = NavigationController(env, mode=args_cli.mode)

    # Reset environment
    obs, _ = env.reset()

    print(f"\n{'='*80}")
    print(f"Interactive ANYmal-C Navigation - Mode: {args_cli.mode.upper()}")
    print(f"{'='*80}\n")

    step_count = 0

    # Main loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get target from controller
            target_x, target_y, target_heading = controller.get_target()

            # Set command in environment
            env.command_manager._terms["pose_command"]._pos_command_w[:, 0] = target_x
            env.command_manager._terms["pose_command"]._pos_command_w[:, 1] = target_y
            env.command_manager._terms["pose_command"]._heading_command_w[:] = target_heading

            # Get action from policy
            if isinstance(obs, dict):
                obs_tensor = obs["policy"]
            else:
                obs_tensor = obs
            actions = policy(obs_tensor)

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1

            # Print status
            if step_count % 100 == 0:
                robot = env.scene["robot"]
                robot_pos = robot.data.root_pos_w[0, :2].cpu().numpy()
                distance = np.linalg.norm(robot_pos - np.array([target_x, target_y]))
                print(f"Step {step_count:5d} | Target: ({target_x:+.2f}, {target_y:+.2f}, {target_heading:+.2f}) | "
                      f"Robot: ({robot_pos[0]:+.2f}, {robot_pos[1]:+.2f}) | Dist: {distance:.2f}m")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
