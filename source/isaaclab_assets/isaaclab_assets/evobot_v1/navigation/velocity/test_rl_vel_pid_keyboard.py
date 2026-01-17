#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical control: RL balance policy + PID velocity controller + keyboard setpoint control.

This script demonstrates a two-level control architecture:
- Inner loop (RL): Pre-trained balance policy controls joint positions to maintain balance
- Outer loop (PID): PID controller generates velocity setpoints from keyboard commands
- User control: Keyboard adjusts desired velocity in real-time

Architecture:
    Keyboard → Velocity Setpoint → PID Controller → Wheel Effort → RL Policy → Joint Actions → Robot

Usage:
    # Run with pre-trained balance policy
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_rl_pid_keyboard.py \
        --load_run 2026-01-15_01-19-39 \
        --checkpoint model_500.pt

    # Run with custom PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_rl_pid_keyboard.py \
        --load_run 2026-01-15_01-19-39 \
        --checkpoint model_500.pt \
        --kp_linear 2.5 --ki_linear 0.15 --kd_linear 0.08

Keyboard Controls:
    - Arrow Up / Numpad 8: Increase forward velocity setpoint
    - Arrow Down / Numpad 2: Decrease forward velocity setpoint
    - Z / Numpad 7: Increase left turn rate (counter-clockwise)
    - X / Numpad 9: Increase right turn rate (clockwise)
    - L: Reset all velocity setpoints to zero
    - ESC: Exit

Control Flow:
    1. User presses keyboard → velocity setpoint updated
    2. PID compares setpoint with actual velocity → computes wheel effort
    3. Wheel effort applied as external force/torque
    4. RL policy observes state → outputs joint actions to maintain balance
    5. Robot executes joint actions → moves and balances
"""

import argparse
import torch
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Hierarchical RL+PID control with keyboard velocity setpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Velocity", help="Task name")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name (e.g., 2026-01-15_01-19-39)")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")
parser.add_argument("--sensitivity", type=float, default=0.3, help="Keyboard command sensitivity")

# PID gains for velocity control
parser.add_argument("--kp_linear", type=float, default=0.10, help="Kp for linear velocity")
parser.add_argument("--ki_linear", type=float, default=0.0, help="Ki for linear velocity")
parser.add_argument("--kd_linear", type=float, default=0.05, help="Kd for linear velocity")
parser.add_argument("--kp_angular", type=float, default=2.0, help="Kp for angular velocity")
parser.add_argument("--ki_angular", type=float, default=0.0, help="Ki for angular velocity")
parser.add_argument("--kd_angular", type=float, default=0.05, help="Kd for angular velocity")

# Wheel parameters
parser.add_argument("--wheel_base", type=float, default=0.2, help="Distance between wheels (m)")
parser.add_argument("--wheel_radius", type=float, default=0.05, help="Wheel radius (m)")
parser.add_argument("--effort_scale", type=float, default=100.0, help="Scale factor for wheel effort")

# Control frequency
parser.add_argument("--pid_decimation", type=int, default=4, help="PID decimation factor (PID freq = RL freq / decimation)")

# Episode settings
parser.add_argument("--episode_length", type=float, default=120.0, help="Episode length in seconds (0 = infinite)")

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
import carb
import omni.appwindow
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


class VelocityKeyboardController:
    """Keyboard controller for velocity setpoint control (no decay).

    Unlike Se2Keyboard which automatically decays to zero, this controller
    maintains the velocity setpoint until a new key is pressed.
    """

    def __init__(self, sensitivity: float = 0.3, device: str = "cuda", sim_app=None):
        """Initialize keyboard controller.

        Args:
            sensitivity: Velocity increment per key press
            device: Device for tensor operations
            sim_app: Simulation app instance for ESC handling
        """
        self.sensitivity = sensitivity
        self.device = device
        self.sim_app = sim_app

        # Current velocity setpoint [vx, wz]
        self.vel_setpoint = torch.zeros(2, device=device)

        # Get carb input interface for polling
        self._input = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = self._appwindow.get_keyboard()

        # Track key press states to debounce
        self._key_pressed = {}

    def update(self):
        """Poll keyboard and update velocity setpoint (call every frame)."""
        # Arrow Up / Numpad 8: Increase forward velocity
        if self._is_key_pressed(carb.input.KeyboardInput.UP) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_8):
            if not self._key_pressed.get("up", False):
                self.vel_setpoint[0] += self.sensitivity
                print(f"\n[↑] Forward velocity: {self.vel_setpoint[0].item():+.2f} m/s")
                self._key_pressed["up"] = True
        else:
            self._key_pressed["up"] = False

        # Arrow Down / Numpad 2: Decrease forward velocity
        if self._is_key_pressed(carb.input.KeyboardInput.DOWN) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_2):
            if not self._key_pressed.get("down", False):
                self.vel_setpoint[0] -= self.sensitivity
                print(f"\n[↓] Forward velocity: {self.vel_setpoint[0].item():+.2f} m/s")
                self._key_pressed["down"] = True
        else:
            self._key_pressed["down"] = False

        # Z / Numpad 7: Turn left
        if self._is_key_pressed(carb.input.KeyboardInput.Z) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_7):
            if not self._key_pressed.get("left", False):
                self.vel_setpoint[1] += self.sensitivity
                print(f"\n[Z] Turn velocity: {self.vel_setpoint[1].item():+.2f} rad/s (LEFT)")
                self._key_pressed["left"] = True
        else:
            self._key_pressed["left"] = False

        # X / Numpad 9: Turn right
        if self._is_key_pressed(carb.input.KeyboardInput.X) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_9):
            if not self._key_pressed.get("right", False):
                self.vel_setpoint[1] -= self.sensitivity
                print(f"\n[X] Turn velocity: {self.vel_setpoint[1].item():+.2f} rad/s (RIGHT)")
                self._key_pressed["right"] = True
        else:
            self._key_pressed["right"] = False

        # L: Reset all velocities
        if self._is_key_pressed(carb.input.KeyboardInput.L):
            if not self._key_pressed.get("reset", False):
                self.vel_setpoint.zero_()
                print(f"\n[L] Reset velocities to zero")
                self._key_pressed["reset"] = True
        else:
            self._key_pressed["reset"] = False

        # ESC: Exit
        if self._is_key_pressed(carb.input.KeyboardInput.ESCAPE):
            if not self._key_pressed.get("esc", False):
                print("\n[ESC] Exiting...")
                if self.sim_app is not None:
                    self.sim_app.close()
                self._key_pressed["esc"] = True
        else:
            self._key_pressed["esc"] = False

    def _is_key_pressed(self, key: carb.input.KeyboardInput) -> bool:
        """Check if a key is currently pressed."""
        return self._input.get_keyboard_value(self._keyboard, key) != 0

    def get_velocity_setpoint(self) -> torch.Tensor:
        """Get current velocity setpoint [vx, wz]."""
        return self.vel_setpoint.clone()

    def reset(self):
        """Reset velocity setpoint to zero."""
        self.vel_setpoint.zero_()


class VelocityPIDController:
    """PID controller for differential drive velocity control.

    Converts linear velocity (vx) and angular velocity (wz) commands into wheel efforts.
    """

    def __init__(
        self,
        kp_linear: float,
        ki_linear: float,
        kd_linear: float,
        kp_angular: float,
        ki_angular: float,
        kd_angular: float,
        wheel_base: float,
        wheel_radius: float,
        effort_scale: float,
        device: str = "cuda",
    ):
        """Initialize PID controller.

        Args:
            kp_linear: Proportional gain for linear velocity
            ki_linear: Integral gain for linear velocity
            kd_linear: Derivative gain for linear velocity
            kp_angular: Proportional gain for angular velocity
            ki_angular: Integral gain for angular velocity
            kd_angular: Derivative gain for angular velocity
            wheel_base: Distance between left and right wheels (m)
            wheel_radius: Wheel radius (m)
            effort_scale: Scale factor for output effort
            device: Device for tensor operations
        """
        self.kp_linear = kp_linear
        self.ki_linear = ki_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.ki_angular = ki_angular
        self.kd_angular = kd_angular

        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.effort_scale = effort_scale
        self.device = device

        # PID state
        self.error_integral_linear = None
        self.error_integral_angular = None
        self.error_prev_linear = None
        self.error_prev_angular = None

    def reset(self, num_envs: int):
        """Reset PID state for given number of environments."""
        self.error_integral_linear = torch.zeros(num_envs, device=self.device)
        self.error_integral_angular = torch.zeros(num_envs, device=self.device)
        self.error_prev_linear = torch.zeros(num_envs, device=self.device)
        self.error_prev_angular = torch.zeros(num_envs, device=self.device)

    def compute(
        self,
        vel_cmd: torch.Tensor,
        vel_current: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute wheel efforts from velocity command and current velocity.

        Args:
            vel_cmd: Desired velocity [vx, wz] (num_envs, 2)
            vel_current: Current velocity [vx, wz] (num_envs, 2)
            dt: Time step (s)

        Returns:
            Wheel efforts [left_effort, right_effort] (num_envs, 2)
        """
        # Extract velocities
        vx_cmd = vel_cmd[:, 0]
        wz_cmd = vel_cmd[:, 1]
        vx_current = vel_current[:, 0]
        wz_current = vel_current[:, 1]

        # Compute errors
        error_linear = vx_cmd - vx_current
        error_angular = wz_cmd - wz_current

        # Update integral
        self.error_integral_linear += error_linear * dt
        self.error_integral_angular += error_angular * dt

        # Compute derivative
        error_derivative_linear = (error_linear - self.error_prev_linear) / dt
        error_derivative_angular = (error_angular - self.error_prev_angular) / dt

        # PID output
        u_linear = (
            self.kp_linear * error_linear
            + self.ki_linear * self.error_integral_linear
            + self.kd_linear * error_derivative_linear
        )
        u_angular = (
            self.kp_angular * error_angular
            + self.ki_angular * self.error_integral_angular
            + self.kd_angular * error_derivative_angular
        )

        # Update previous errors
        self.error_prev_linear = error_linear.clone()
        self.error_prev_angular = error_angular.clone()

        # Convert to wheel velocities (differential drive kinematics)
        # v_left = v_x - (L/2) * omega_z
        # v_right = v_x + (L/2) * omega_z
        v_left = u_linear - (self.wheel_base / 2.0) * u_angular
        v_right = u_linear + (self.wheel_base / 2.0) * u_angular

        # Convert to wheel efforts (simplified: effort proportional to desired velocity)
        effort_left = v_left * self.effort_scale
        effort_right = v_right * self.effort_scale

        # Stack into (num_envs, 2)
        wheel_efforts = torch.stack([effort_left, effort_right], dim=-1)

        return wheel_efforts


def print_keyboard_help():
    """Print keyboard control instructions."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL CONTROL: RL BALANCE + PID VELOCITY + KEYBOARD SETPOINT")
    print("=" * 80)
    print("Velocity Setpoint Control:")
    print("  Arrow Up / Numpad 8   : Increase forward velocity")
    print("  Arrow Down / Numpad 2 : Decrease forward velocity")
    print("  Z / Numpad 7          : Turn left (counter-clockwise)")
    print("  X / Numpad 9          : Turn right (clockwise)")
    print("  L                     : Reset all velocities to zero")
    print("\nUtility:")
    print("  ESC                   : Exit")
    print("\nControl Architecture:")
    print("  1. Keyboard → Velocity Setpoint")
    print("  2. PID Controller → Wheel Effort (from velocity error)")
    print("  3. RL Policy → Joint Actions (for balance)")
    print("  4. Robot → Execute actions and move")
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

    # Override episode length if specified
    if args_cli.episode_length > 0:
        env_cfg.episode_length_s = args_cli.episode_length

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print environment info
    print("\n" + "=" * 80)
    print("EVOBOT V1 - HIERARCHICAL RL+PID CONTROL")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Episode length: {env_cfg.episode_length_s:.1f} seconds")
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
        print("[ERROR] This script requires a trained balance policy.")
        print("[INFO] Usage: --load_run <run_name> --checkpoint model_500.pt\n")
        env.close()
        return

    # Initialize PID controller
    print("\n" + "=" * 80)
    print("PID CONTROLLER CONFIGURATION")
    print("=" * 80)
    print(f"Linear PID:  Kp={args_cli.kp_linear}, Ki={args_cli.ki_linear}, Kd={args_cli.kd_linear}")
    print(f"Angular PID: Kp={args_cli.kp_angular}, Ki={args_cli.ki_angular}, Kd={args_cli.kd_angular}")
    print(f"Wheel base: {args_cli.wheel_base} m")
    print(f"Wheel radius: {args_cli.wheel_radius} m")
    print(f"Effort scale: {args_cli.effort_scale}")
    print("=" * 80)

    pid_controller = VelocityPIDController(
        kp_linear=args_cli.kp_linear,
        ki_linear=args_cli.ki_linear,
        kd_linear=args_cli.kd_linear,
        kp_angular=args_cli.kp_angular,
        ki_angular=args_cli.ki_angular,
        kd_angular=args_cli.kd_angular,
        wheel_base=args_cli.wheel_base,
        wheel_radius=args_cli.wheel_radius,
        effort_scale=args_cli.effort_scale,
        device=args_cli.device,
    )
    pid_controller.reset(args_cli.num_envs)

    # Initialize keyboard controller for velocity setpoints (no decay)
    keyboard = VelocityKeyboardController(
        sensitivity=args_cli.sensitivity,
        device=args_cli.device,
        sim_app=simulation_app,
    )

    # Print keyboard help
    print_keyboard_help()

    # Reset environment
    env.reset()
    obs = env_wrapped.get_observations()
    print("[INFO] Environment ready. Control velocity setpoint with keyboard...\n")

    # Get robot asset
    robot = env.unwrapped.scene["robot"]

    # Get wheel joint indices (assuming last 2 joints are wheels)
    # Adjust these indices based on your robot's joint order
    wheel_joint_names = ["left_wheel_joint", "right_wheel_joint"]  # Left, Right wheels
    wheel_indices = []
    for name in wheel_joint_names:
        try:
            idx = robot.joint_names.index(name)
            wheel_indices.append(idx)
        except ValueError:
            print(f"[WARNING] Wheel joint '{name}' not found. Available joints: {robot.joint_names}")

    if len(wheel_indices) != 2:
        print("[ERROR] Could not find both wheel joints. Exiting.")
        env.close()
        return

    print(f"[INFO] Wheel joint indices: {wheel_indices} (joints: {wheel_joint_names})")

    # Get simulation timestep for RL control loop
    dt_rl = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation

    # PID control timestep (slower than RL - runs every pid_decimation steps)
    dt_pid = dt_rl * args_cli.pid_decimation

    # Print control frequencies
    rl_freq = 1.0 / dt_rl
    pid_freq = 1.0 / dt_pid
    print(f"\n[INFO] Control Frequencies:")
    print(f"  RL Policy:      {rl_freq:.1f} Hz (dt={dt_rl:.4f}s)")
    print(f"  PID Controller: {pid_freq:.1f} Hz (dt={dt_pid:.4f}s)")
    print(f"  PID Decimation: {args_cli.pid_decimation}x slower than RL")
    print()

    # Main loop
    count = 0
    pid_step_counter = 0
    wheel_efforts = torch.zeros(args_cli.num_envs, 2, device=args_cli.device)  # Initialize wheel efforts
    vel_current = torch.zeros(args_cli.num_envs, 2, device=args_cli.device)  # Initialize current velocity

    while simulation_app.is_running():
        with torch.inference_mode():
            # Update keyboard state and get velocity setpoint [vx, wz] (no decay - maintains setpoint)
            keyboard.update()
            vel_setpoint = keyboard.get_velocity_setpoint()

            # Always read current velocity (for display)
            vel_current = torch.stack([
                robot.data.root_lin_vel_b[:, 0],  # vx in base frame
                robot.data.root_ang_vel_b[:, 2],  # wz in base frame
            ], dim=-1)

            # PID controller runs at lower frequency (every pid_decimation steps)
            if pid_step_counter % args_cli.pid_decimation == 0:
                # Expand setpoint to all environments (shape: [num_envs, 2])
                vel_setpoint_expanded = vel_setpoint.unsqueeze(0).expand(args_cli.num_envs, -1)

                # Compute PID wheel efforts (using PID timestep)
                wheel_efforts = pid_controller.compute(vel_setpoint_expanded, vel_current, dt_pid)

            pid_step_counter += 1

            # Get action from trained RL policy (for balance)
            actions = policy(obs)

            # Apply wheel efforts as external forces/torques to wheel joints
            # Create effort tensor for all joints (zero for non-wheel joints)
            joint_efforts = torch.zeros(args_cli.num_envs, robot.num_joints, device=args_cli.device)
            joint_efforts[:, wheel_indices[0]] = wheel_efforts[:, 0]  # Left wheel
            joint_efforts[:, wheel_indices[1]] = wheel_efforts[:, 1]  # Right wheel

            # Set joint efforts (this applies wheel efforts on top of RL policy actions)
            robot.set_joint_effort_target(joint_efforts)
            robot.write_data_to_sim()

            # Step environment (RL policy actions for balance + PID wheel efforts for velocity)
            obs, reward, dones, _ = env_wrapped.step(actions)

            # Print status every 50 steps
            count += 1
            if count % 50 == 0:
                # Print current velocity setpoint and actual velocity
                vx_setpoint = vel_setpoint[0].item()  # vel_setpoint is 1D [vx, wz]
                wz_setpoint = vel_setpoint[1].item()
                vx_current = vel_current[0, 0].item()  # vel_current is 2D [num_envs, 2]
                wz_current = vel_current[0, 1].item()

                print(f"\r[SETPOINT] vx: {vx_setpoint:+.2f} m/s | ωz: {wz_setpoint:+.2f} rad/s", end="")
                print(f" | [ACTUAL] vx: {vx_current:+.2f} m/s | ωz: {wz_current:+.2f} rad/s", end="")

                if torch.is_tensor(reward):
                    mean_reward = reward.mean().item()
                else:
                    mean_reward = float(reward)
                print(f" | Reward: {mean_reward:+.3f}")

            # Handle resets (wrapped env uses 'dones' instead of terminated/truncated)
            if torch.is_tensor(dones) and dones.any():
                print("\n[RESET] Environment terminated/truncated. Resetting...")
                # Reset PID state for terminated environments
                reset_ids = torch.where(dones)[0]
                pid_controller.error_integral_linear[reset_ids] = 0.0
                pid_controller.error_integral_angular[reset_ids] = 0.0
                pid_controller.error_prev_linear[reset_ids] = 0.0
                pid_controller.error_prev_angular[reset_ids] = 0.0

                # CRITICAL: Reset wheel efforts to zero (prevent jerky motion after reset)
                wheel_efforts[reset_ids] = 0.0

                # Get fresh observations
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
