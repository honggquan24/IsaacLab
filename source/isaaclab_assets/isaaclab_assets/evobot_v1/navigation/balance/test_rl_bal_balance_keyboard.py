#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical control: RL balance policy + PID angle controller + keyboard setpoint control.

This script demonstrates a two-level control architecture for BALANCE task:
- Inner loop (RL): Pre-trained balance policy controls joint efforts to maintain balance
- Outer loop (PID): PID controller generates wheel efforts from pitch/roll angle commands
- User control: Keyboard adjusts desired pitch/roll angles in real-time

Architecture:
    Keyboard → Pitch/Roll Angle Setpoint → PID Controller → Wheel Effort → RL Policy → Joint Actions → Robot

Key Difference from Velocity Control:
    - Velocity: PID tracks velocity (vx, wz) → generates wheel efforts
    - Balance: PID tracks angles (pitch, roll) → generates wheel efforts to correct tilt

Usage:
    # Run with pre-trained balance policy
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_rl_bal_balance_keyboard.py \
        --load_run 2026-01-11_17-15-59 \
        --checkpoint model_500.pt

    # Run with custom PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_rl_bal_balance_keyboard.py \
        --load_run 2026-01-11_17-15-59 \
        --checkpoint model_500.pt \
        --kp_pitch 50.0 --kd_pitch 10.0 --kp_roll 50.0 --kd_roll 10.0

Keyboard Controls:
    - Arrow Up / Numpad 8: Increase pitch angle (lean forward)
    - Arrow Down / Numpad 2: Decrease pitch angle (lean backward)
    - Arrow Left / Numpad 4: Decrease roll angle (lean left)
    - Arrow Right / Numpad 6: Increase roll angle (lean right)
    - L: Reset to upright (pitch=0, roll=0)
    - ESC: Exit

Control Flow:
    1. User presses keyboard → pitch/roll angle setpoint updated
    2. PID compares setpoint with actual angle (from IMU) → computes wheel effort
    3. Wheel effort applied to wheel joints
    4. RL policy observes state → outputs joint actions to maintain balance
    5. Robot executes joint actions → balances at desired angle
"""

import argparse
import torch
import os
import math

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Hierarchical RL+PID balance control with keyboard angle setpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Balance", help="Task name")
parser.add_argument("--load_run", type=str, default=None, help="Run directory name (e.g., 2026-01-11_17-15-59)")
parser.add_argument("--checkpoint", type=str, default="model_500.pt", help="Checkpoint filename")
parser.add_argument("--sensitivity", type=float, default=0.05, help="Keyboard command sensitivity (radians)")

# PID gains for angle control
parser.add_argument("--kp_pitch", type=float, default=50.0, help="Kp for pitch angle control")
parser.add_argument("--ki_pitch", type=float, default=0.0, help="Ki for pitch angle control")
parser.add_argument("--kd_pitch", type=float, default=10.0, help="Kd for pitch angle control")
parser.add_argument("--kp_roll", type=float, default=50.0, help="Kp for roll angle control")
parser.add_argument("--ki_roll", type=float, default=0.0, help="Ki for roll angle control")
parser.add_argument("--kd_roll", type=float, default=10.0, help="Kd for roll angle control")

# Wheel parameters
parser.add_argument("--wheel_base", type=float, default=0.2, help="Distance between wheels (m)")
parser.add_argument("--effort_scale", type=float, default=100.0, help="Scale factor for wheel effort")

# Control frequency
parser.add_argument("--pid_decimation", type=int, default=4, help="PID decimation factor (PID freq = RL freq / decimation)")

# Episode settings
parser.add_argument("--episode_length", type=float, default=120.0, help="Episode length in seconds")

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


class BalanceKeyboardController:
    """Keyboard controller for pitch/roll angle setpoint control (no decay).

    Controls desired tilt angles for balance task.
    """

    def __init__(self, sensitivity: float = 0.05, device: str = "cuda", sim_app=None):
        """Initialize keyboard controller.

        Args:
            sensitivity: Angle increment per key press (radians)
            device: Device for tensor operations
            sim_app: Simulation app instance for ESC handling
        """
        self.sensitivity = sensitivity
        self.device = device
        self.sim_app = sim_app

        # Current angle setpoint [pitch, roll] in radians
        self.angle_setpoint = torch.zeros(2, device=device)

        # Get carb input interface for polling
        self._input = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = self._appwindow.get_keyboard()

        # Track key press states to debounce
        self._key_pressed = {}

    def update(self):
        """Poll keyboard and update angle setpoint (call every frame)."""
        # Arrow Up / Numpad 8: Increase pitch (lean forward)
        if self._is_key_pressed(carb.input.KeyboardInput.UP) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_8):
            if not self._key_pressed.get("up", False):
                self.angle_setpoint[0] += self.sensitivity
                print(f"\n[↑] Pitch: {math.degrees(self.angle_setpoint[0].item()):+.2f}° (FORWARD)")
                self._key_pressed["up"] = True
        else:
            self._key_pressed["up"] = False

        # Arrow Down / Numpad 2: Decrease pitch (lean backward)
        if self._is_key_pressed(carb.input.KeyboardInput.DOWN) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_2):
            if not self._key_pressed.get("down", False):
                self.angle_setpoint[0] -= self.sensitivity
                print(f"\n[↓] Pitch: {math.degrees(self.angle_setpoint[0].item()):+.2f}° (BACKWARD)")
                self._key_pressed["down"] = True
        else:
            self._key_pressed["down"] = False

        # Arrow Left / Numpad 4: Decrease roll (lean left)
        if self._is_key_pressed(carb.input.KeyboardInput.LEFT) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_4):
            if not self._key_pressed.get("left", False):
                self.angle_setpoint[1] -= self.sensitivity
                print(f"\n[←] Roll: {math.degrees(self.angle_setpoint[1].item()):+.2f}° (LEFT)")
                self._key_pressed["left"] = True
        else:
            self._key_pressed["left"] = False

        # Arrow Right / Numpad 6: Increase roll (lean right)
        if self._is_key_pressed(carb.input.KeyboardInput.RIGHT) or self._is_key_pressed(carb.input.KeyboardInput.NUMPAD_6):
            if not self._key_pressed.get("right", False):
                self.angle_setpoint[1] += self.sensitivity
                print(f"\n[→] Roll: {math.degrees(self.angle_setpoint[1].item()):+.2f}° (RIGHT)")
                self._key_pressed["right"] = True
        else:
            self._key_pressed["right"] = False

        # L: Reset to upright
        if self._is_key_pressed(carb.input.KeyboardInput.L):
            if not self._key_pressed.get("reset", False):
                self.angle_setpoint.zero_()
                print(f"\n[L] Reset to upright (pitch=0°, roll=0°)")
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

    def get_angle_setpoint(self) -> torch.Tensor:
        """Get current angle setpoint [pitch, roll] in radians."""
        return self.angle_setpoint.clone()

    def reset(self):
        """Reset angle setpoint to zero (upright)."""
        self.angle_setpoint.zero_()


class BalancePIDController:
    """PID controller for balance angle control.

    Converts pitch/roll angle commands into wheel efforts to maintain balance.
    """

    def __init__(
        self,
        kp_pitch: float,
        ki_pitch: float,
        kd_pitch: float,
        kp_roll: float,
        ki_roll: float,
        kd_roll: float,
        wheel_base: float,
        effort_scale: float,
        device: str = "cuda",
    ):
        """Initialize PID controller.

        Args:
            kp_pitch: Proportional gain for pitch angle
            ki_pitch: Integral gain for pitch angle
            kd_pitch: Derivative gain for pitch angle
            kp_roll: Proportional gain for roll angle
            ki_roll: Integral gain for roll angle
            kd_roll: Derivative gain for roll angle
            wheel_base: Distance between left and right wheels (m)
            effort_scale: Scale factor for output effort
            device: Device for tensor operations
        """
        self.kp_pitch = kp_pitch
        self.ki_pitch = ki_pitch
        self.kd_pitch = kd_pitch
        self.kp_roll = kp_roll
        self.ki_roll = ki_roll
        self.kd_roll = kd_roll

        self.wheel_base = wheel_base
        self.effort_scale = effort_scale
        self.device = device

        # PID state
        self.error_integral_pitch = None
        self.error_integral_roll = None
        self.error_prev_pitch = None
        self.error_prev_roll = None

    def reset(self, num_envs: int):
        """Reset PID state for given number of environments."""
        self.error_integral_pitch = torch.zeros(num_envs, device=self.device)
        self.error_integral_roll = torch.zeros(num_envs, device=self.device)
        self.error_prev_pitch = torch.zeros(num_envs, device=self.device)
        self.error_prev_roll = torch.zeros(num_envs, device=self.device)

    def compute(
        self,
        angle_cmd: torch.Tensor,
        angle_current: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute wheel efforts from angle command and current angle.

        Args:
            angle_cmd: Desired angle [pitch, roll] in radians (num_envs, 2)
            angle_current: Current angle [pitch, roll] in radians (num_envs, 2)
            dt: Time step (s)

        Returns:
            Wheel efforts [left_effort, right_effort] (num_envs, 2)
        """
        # Extract angles
        pitch_cmd = angle_cmd[:, 0]
        roll_cmd = angle_cmd[:, 1]
        pitch_current = angle_current[:, 0]
        roll_current = angle_current[:, 1]

        # Compute errors
        error_pitch = pitch_cmd - pitch_current
        error_roll = roll_cmd - roll_current

        # Update integral
        self.error_integral_pitch += error_pitch * dt
        self.error_integral_roll += error_roll * dt

        # Compute derivative
        error_derivative_pitch = (error_pitch - self.error_prev_pitch) / dt
        error_derivative_roll = (error_roll - self.error_prev_roll) / dt

        # PID output
        u_pitch = (
            self.kp_pitch * error_pitch
            + self.ki_pitch * self.error_integral_pitch
            + self.kd_pitch * error_derivative_pitch
        )
        u_roll = (
            self.kp_roll * error_roll
            + self.ki_roll * self.error_integral_roll
            + self.kd_roll * error_derivative_roll
        )

        # Update previous errors
        self.error_prev_pitch = error_pitch.clone()
        self.error_prev_roll = error_roll.clone()

        # Convert to wheel efforts
        # Pitch control: Both wheels same direction (forward/backward)
        # Roll control: Wheels opposite direction (turn left/right)
        effort_left = (u_pitch - u_roll) * self.effort_scale
        effort_right = (u_pitch + u_roll) * self.effort_scale

        # Stack into (num_envs, 2)
        wheel_efforts = torch.stack([effort_left, effort_right], dim=-1)

        return wheel_efforts


def quaternion_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion [w, x, y, z] (num_envs, 4)

    Returns:
        Euler angles [roll, pitch, yaw] in radians (num_envs, 3)
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def print_keyboard_help():
    """Print keyboard control instructions."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL CONTROL: RL BALANCE + PID ANGLE + KEYBOARD SETPOINT")
    print("=" * 80)
    print("Angle Setpoint Control:")
    print("  Arrow Up / Numpad 8   : Increase pitch (lean forward)")
    print("  Arrow Down / Numpad 2 : Decrease pitch (lean backward)")
    print("  Arrow Left / Numpad 4 : Decrease roll (lean left)")
    print("  Arrow Right / Numpad 6: Increase roll (lean right)")
    print("  L                     : Reset to upright (pitch=0, roll=0)")
    print("\nUtility:")
    print("  ESC                   : Exit")
    print("\nControl Architecture:")
    print("  1. Keyboard → Pitch/Roll Angle Setpoint")
    print("  2. PID Controller → Wheel Effort (from angle error)")
    print("  3. RL Policy → Joint Actions (for balance)")
    print("  4. Robot → Execute actions and balance")
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
    print("EVOBOT V1 - HIERARCHICAL RL+PID BALANCE CONTROL")
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
    print(f"Pitch PID: Kp={args_cli.kp_pitch}, Ki={args_cli.ki_pitch}, Kd={args_cli.kd_pitch}")
    print(f"Roll PID:  Kp={args_cli.kp_roll}, Ki={args_cli.ki_roll}, Kd={args_cli.kd_roll}")
    print(f"Wheel base: {args_cli.wheel_base} m")
    print(f"Effort scale: {args_cli.effort_scale}")
    print("=" * 80)

    pid_controller = BalancePIDController(
        kp_pitch=args_cli.kp_pitch,
        ki_pitch=args_cli.ki_pitch,
        kd_pitch=args_cli.kd_pitch,
        kp_roll=args_cli.kp_roll,
        ki_roll=args_cli.ki_roll,
        kd_roll=args_cli.kd_roll,
        wheel_base=args_cli.wheel_base,
        effort_scale=args_cli.effort_scale,
        device=args_cli.device,
    )
    pid_controller.reset(args_cli.num_envs)

    # Initialize keyboard controller for angle setpoints (no decay)
    keyboard = BalanceKeyboardController(
        sensitivity=args_cli.sensitivity,
        device=args_cli.device,
        sim_app=simulation_app,
    )

    # Print keyboard help
    print_keyboard_help()

    # Reset environment
    env.reset()
    obs = env_wrapped.get_observations()
    print("[INFO] Environment ready. Control angle setpoint with keyboard...\n")

    # Get robot asset
    robot = env.unwrapped.scene["robot"]

    # Get IMU sensor
    imu = env.unwrapped.scene["imu"]

    # Get wheel joint indices
    wheel_joint_names = ["left_wheel_joint", "right_wheel_joint"]
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
    wheel_efforts = torch.zeros(args_cli.num_envs, 2, device=args_cli.device)
    angles_current = torch.zeros(args_cli.num_envs, 2, device=args_cli.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            # Update keyboard state and get angle setpoint [pitch, roll] (no decay)
            keyboard.update()
            angle_setpoint = keyboard.get_angle_setpoint()

            # Read current pitch/roll from IMU quaternion
            quat = imu.data.quat_w  # Shape: (num_envs, 4) [w, x, y, z]
            euler_angles = quaternion_to_euler(quat)  # Shape: (num_envs, 3) [roll, pitch, yaw]
            angles_current = torch.stack([
                euler_angles[:, 1],  # pitch
                euler_angles[:, 0],  # roll
            ], dim=-1)

            # PID controller runs at lower frequency (every pid_decimation steps)
            if pid_step_counter % args_cli.pid_decimation == 0:
                # Expand setpoint to all environments (shape: [num_envs, 2])
                angle_setpoint_expanded = angle_setpoint.unsqueeze(0).expand(args_cli.num_envs, -1)

                # Compute PID wheel efforts (using PID timestep)
                wheel_efforts = pid_controller.compute(angle_setpoint_expanded, angles_current, dt_pid)

            pid_step_counter += 1

            # Get action from trained RL policy (for balance)
            actions = policy(obs)

            # Apply wheel efforts to wheel joints
            # Create effort tensor for all joints (zero for non-wheel joints)
            joint_efforts = torch.zeros(args_cli.num_envs, robot.num_joints, device=args_cli.device)
            joint_efforts[:, wheel_indices[0]] = wheel_efforts[:, 0]  # Left wheel
            joint_efforts[:, wheel_indices[1]] = wheel_efforts[:, 1]  # Right wheel

            # Set joint efforts (this applies wheel efforts on top of RL policy actions)
            robot.set_joint_effort_target(joint_efforts)
            robot.write_data_to_sim()

            # Step environment (RL policy actions for balance + PID wheel efforts for angle control)
            obs, reward, dones, _ = env_wrapped.step(actions)

            # Print status every 50 steps
            count += 1
            if count % 50 == 0:
                # Print current angle setpoint and actual angle
                pitch_setpoint = math.degrees(angle_setpoint[0].item())
                roll_setpoint = math.degrees(angle_setpoint[1].item())
                pitch_current = math.degrees(angles_current[0, 0].item())
                roll_current = math.degrees(angles_current[0, 1].item())

                print(f"\r[SETPOINT] pitch: {pitch_setpoint:+.2f}° | roll: {roll_setpoint:+.2f}°", end="")
                print(f" | [ACTUAL] pitch: {pitch_current:+.2f}° | roll: {roll_current:+.2f}°", end="")

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
                pid_controller.error_integral_pitch[reset_ids] = 0.0
                pid_controller.error_integral_roll[reset_ids] = 0.0
                pid_controller.error_prev_pitch[reset_ids] = 0.0
                pid_controller.error_prev_roll[reset_ids] = 0.0

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
