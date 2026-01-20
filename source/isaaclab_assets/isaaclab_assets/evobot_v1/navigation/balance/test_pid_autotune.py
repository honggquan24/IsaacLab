#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auto-tuning PID controller for Evobot V1 balance using Twiddle algorithm.

This script automatically finds optimal PID gains using coordinate ascent optimization.
It evaluates different gain combinations and selects the one with best performance.

Algorithm: Twiddle (Coordinate Ascent)
- Start with initial gains [Kp, Ki, Kd]
- Try increasing/decreasing each gain
- Keep changes that improve cost function
- Reduce step size when no improvement found

Cost function: Minimize squared error + control effort
    J = ∫(error² + λ*action²)dt

Usage:
    # Run auto-tuning with default settings
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_autotune.py

    # Run with custom initial gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_autotune.py \
        --init_kp 0.5 --init_ki 0.01 --init_kd 0.2

    # Run with custom tuning parameters
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/balance/test_pid_autotune.py \
        --max_iterations 20 --tolerance 0.001

Output:
    - Console: Real-time tuning progress and best gains found
    - CSV file: logs/pid_autotune_<timestamp>.csv
"""

import argparse
import torch
import os
import csv
from datetime import datetime
from typing import Tuple

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Auto-tuning PID controller using Twiddle algorithm")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments for tuning")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Balance", help="Task name")

# Initial PID gains
parser.add_argument("--init_kp", type=float, default=0.5, help="Initial Kp")
parser.add_argument("--init_ki", type=float, default=0.01, help="Initial Ki")
parser.add_argument("--init_kd", type=float, default=0.2, help="Initial Kd")

# Twiddle parameters
parser.add_argument("--max_iterations", type=int, default=50, help="Max tuning iterations")
parser.add_argument("--tolerance", type=float, default=0.01, help="Convergence tolerance (sum of all step sizes)")
parser.add_argument("--init_step", type=float, default=0.05, help="Initial step size for tuning (relative to param)")
parser.add_argument("--step_scale", type=float, default=1.05, help="Step size increase factor (smaller = more stable)")
parser.add_argument("--step_shrink", type=float, default=0.95, help="Step size decrease factor (closer to 1 = faster convergence)")

# Evaluation parameters
parser.add_argument("--eval_steps", type=int, default=600, help="Steps per evaluation (10 seconds at 60Hz for better stability)")
parser.add_argument("--cost_weight_action", type=float, default=0.01, help="Weight for action cost (lower = focus more on error)")

# Control parameters
parser.add_argument("--effort_scale", type=float, default=1.0, help="Scale factor for wheel effort")

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
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat


class BalancePIDController:
    """PID controller with tunable gains."""

    def __init__(
        self,
        kp_roll: float,
        ki_roll: float,
        kd_roll: float,
        kp_pitch: float,
        ki_pitch: float,
        kd_pitch: float,
        effort_scale: float,
        num_envs: int,
        device: str = "cuda",
    ):
        self.kp_roll = kp_roll
        self.ki_roll = ki_roll
        self.kd_roll = kd_roll
        self.kp_pitch = kp_pitch
        self.ki_pitch = ki_pitch
        self.kd_pitch = kd_pitch
        self.effort_scale = effort_scale
        self.num_envs = num_envs
        self.device = device

        # PID state
        self.roll_integral = torch.zeros(num_envs, device=device)
        self.roll_prev_error = torch.zeros(num_envs, device=device)
        self.pitch_integral = torch.zeros(num_envs, device=device)
        self.pitch_prev_error = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor = None):
        """Reset PID state."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.roll_integral[env_ids] = 0.0
        self.roll_prev_error[env_ids] = 0.0
        self.pitch_integral[env_ids] = 0.0
        self.pitch_prev_error[env_ids] = 0.0

    def update_gains(
        self,
        kp_roll: float = None,
        ki_roll: float = None,
        kd_roll: float = None,
        kp_pitch: float = None,
        ki_pitch: float = None,
        kd_pitch: float = None,
    ):
        """Update PID gains."""
        if kp_roll is not None:
            self.kp_roll = kp_roll
        if ki_roll is not None:
            self.ki_roll = ki_roll
        if kd_roll is not None:
            self.kd_roll = kd_roll
        if kp_pitch is not None:
            self.kp_pitch = kp_pitch
        if ki_pitch is not None:
            self.ki_pitch = ki_pitch
        if kd_pitch is not None:
            self.kd_pitch = kd_pitch

    def compute(
        self,
        current_rpy: torch.Tensor,
        target_rpy: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute control actions."""
        # Extract angles
        roll_current = current_rpy[:, 0]
        pitch_current = current_rpy[:, 1]
        roll_target = target_rpy[:, 0]
        pitch_target = target_rpy[:, 1]

        # Compute errors
        roll_error = roll_target - roll_current
        pitch_error = pitch_target - pitch_current

        # Roll PID
        self.roll_integral += roll_error * dt
        self.roll_integral = torch.clamp(self.roll_integral, -10.0, 10.0)
        roll_derivative = (roll_error - self.roll_prev_error) / dt
        roll_output = (
            self.kp_roll * roll_error + self.ki_roll * self.roll_integral + self.kd_roll * roll_derivative
        )
        self.roll_prev_error = roll_error.clone()

        # Pitch PID
        self.pitch_integral += pitch_error * dt
        self.pitch_integral = torch.clamp(self.pitch_integral, -10.0, 10.0)
        pitch_derivative = (pitch_error - self.pitch_prev_error) / dt
        pitch_output = (
            self.kp_pitch * pitch_error + self.ki_pitch * self.pitch_integral + self.kd_pitch * pitch_derivative
        )
        self.pitch_prev_error = pitch_error.clone()

        # Convert to wheel commands
        wheel_left = (pitch_output + roll_output) * self.effort_scale
        wheel_right = (pitch_output - roll_output) * self.effort_scale
        arm_effort = torch.zeros(self.num_envs, device=self.device)
        gripper_left = torch.zeros(self.num_envs, device=self.device)
        gripper_right = torch.zeros(self.num_envs, device=self.device)

        actions = torch.stack([wheel_left, wheel_right, arm_effort, gripper_left, gripper_right], dim=-1)
        actions = torch.clamp(actions, -1.0, 1.0)

        return actions


class PIDAutoTuner:
    """Automatic PID tuning using Twiddle (Coordinate Ascent) algorithm."""

    def __init__(
        self,
        env,
        robot,
        init_gains: list[float],
        max_iterations: int = 30,
        tolerance: float = 0.001,
        init_step: float = 0.1,
        step_scale: float = 1.1,
        step_shrink: float = 0.9,
        eval_steps: int = 300,
        cost_weight_action: float = 0.1,
        dt: float = 1 / 60,
        device: str = "cuda",
    ):
        """Initialize auto-tuner.

        Args:
            env: Isaac Lab environment
            robot: Robot articulation
            init_gains: Initial gains [Kp_roll, Ki_roll, Kd_roll, Kp_pitch, Ki_pitch, Kd_pitch]
            max_iterations: Maximum tuning iterations
            tolerance: Convergence threshold for step sizes
            init_step: Initial step size for each parameter
            step_scale: Factor to increase step when improvement found
            step_shrink: Factor to decrease step when no improvement
            eval_steps: Number of simulation steps per evaluation
            cost_weight_action: Weight for action cost in objective
            dt: Control timestep
            device: Torch device
        """
        self.env = env
        self.robot = robot
        self.gains = init_gains.copy()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.eval_steps = eval_steps
        self.cost_weight_action = cost_weight_action
        self.dt = dt
        self.device = device

        # Twiddle step sizes (proportional to initial gain values to avoid huge steps on small gains)
        self.steps = [max(init_step * abs(g), init_step * 0.01) if g != 0 else init_step * 0.01 for g in init_gains]
        self.step_scale = step_scale
        self.step_shrink = step_shrink

        # Best cost found
        self.best_cost = float("inf")
        self.best_gains = init_gains.copy()

        # History for logging
        self.history = []

        # Track iterations without improvement for early stopping
        self.no_improvement_count = 0
        self.max_no_improvement = 10

    def evaluate_gains(self, gains: list[float]) -> float:
        """Evaluate cost function for given gains.

        Cost = mean(error²) + weight * mean(action²)

        Args:
            gains: [Kp_roll, Ki_roll, Kd_roll, Kp_pitch, Ki_pitch, Kd_pitch]

        Returns:
            Cost value (lower is better)
        """
        # Create controller with current gains
        controller = BalancePIDController(
            kp_roll=gains[0],
            ki_roll=gains[1],
            kd_roll=gains[2],
            kp_pitch=gains[3],
            ki_pitch=gains[4],
            kd_pitch=gains[5],
            effort_scale=args_cli.effort_scale,
            num_envs=self.env.unwrapped.num_envs,
            device=self.device,
        )

        # Reset environment (MUST be outside inference_mode!)
        self.env.reset()
        controller.reset()

        # Target orientation (upright)
        target_rpy = torch.zeros(self.env.unwrapped.num_envs, 3, device=self.device)

        # Accumulate cost
        total_error_squared = 0.0
        total_action_squared = 0.0
        num_steps = 0

        # Run evaluation
        for _ in range(self.eval_steps):
            with torch.inference_mode():
                # Get current orientation
                quat = self.robot.data.root_quat_w
                roll, pitch, yaw = euler_xyz_from_quat(quat)
                current_rpy = torch.stack([roll, pitch, yaw], dim=-1)

                # Compute control
                actions = controller.compute(current_rpy, target_rpy, self.dt)

            # Step environment (must be outside inference_mode for env.reset() calls)
            obs, reward, terminated, truncated, info = self.env.step(actions)

            with torch.inference_mode():
                # Compute cost components
                error_rp = current_rpy[:, :2] - target_rpy[:, :2]  # Roll and pitch errors
                error_squared = (error_rp**2).sum(dim=-1).mean()
                action_squared = (actions[:, :2] ** 2).sum(dim=-1).mean()  # Only wheel actions

                total_error_squared += error_squared.item()
                total_action_squared += action_squared.item()
                num_steps += 1

                # Early termination if robot falls
                if terminated.any():
                    # Penalty for falling
                    total_error_squared += 100.0 * (self.eval_steps - num_steps)
                    break

        # Compute average cost
        avg_error_squared = total_error_squared / num_steps
        avg_action_squared = total_action_squared / num_steps
        cost = avg_error_squared + self.cost_weight_action * avg_action_squared

        return cost

    def tune(self) -> Tuple[list[float], float]:
        """Run Twiddle algorithm to find optimal gains.

        Returns:
            Tuple of (best_gains, best_cost)
        """
        print("\n" + "=" * 80)
        print("PID AUTO-TUNING USING TWIDDLE ALGORITHM")
        print("=" * 80)
        print(f"Initial gains: Kp_roll={self.gains[0]:.3f}, Ki_roll={self.gains[1]:.3f}, Kd_roll={self.gains[2]:.3f}")
        print(f"              Kp_pitch={self.gains[3]:.3f}, Ki_pitch={self.gains[4]:.3f}, Kd_pitch={self.gains[5]:.3f}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Tolerance: {self.tolerance}")
        print("=" * 80 + "\n")

        # Evaluate initial gains
        self.best_cost = self.evaluate_gains(self.gains)
        self.best_gains = self.gains.copy()
        print(f"[Iteration 0] Initial cost: {self.best_cost:.6f}")
        self.history.append({"iteration": 0, "cost": self.best_cost, "gains": self.gains.copy()})

        iteration = 0

        while sum(self.steps) > self.tolerance and iteration < self.max_iterations:
            iteration += 1
            print(f"\n[Iteration {iteration}] Best cost: {self.best_cost:.6f}")
            print(f"  Step sizes sum: {sum(self.steps):.6f}, Steps: {[f'{s:.4f}' for s in self.steps]}")

            iteration_improved = False

            for i in range(len(self.gains)):
                # Store original value
                original_gain = self.best_gains[i]

                # Try increasing gain
                test_gains = self.best_gains.copy()
                test_gains[i] = original_gain + self.steps[i]
                test_gains[i] = max(0.0, test_gains[i])  # Ensure non-negative

                cost_increase = self.evaluate_gains(test_gains)
                print(f"  Param {i} ({original_gain:.4f}): +{self.steps[i]:.4f} → {test_gains[i]:.4f}, cost={cost_increase:.6f}", end="")

                if cost_increase < self.best_cost:
                    # Improvement found by increasing
                    self.best_cost = cost_increase
                    self.best_gains = test_gains.copy()
                    self.steps[i] *= self.step_scale
                    iteration_improved = True
                    print(" ✓ BETTER")
                else:
                    # Try decreasing gain
                    test_gains[i] = original_gain - self.steps[i]
                    test_gains[i] = max(0.0, test_gains[i])

                    cost_decrease = self.evaluate_gains(test_gains)
                    print(f" | -{self.steps[i]:.4f} → {test_gains[i]:.4f}, cost={cost_decrease:.6f}", end="")

                    if cost_decrease < self.best_cost:
                        # Improvement found by decreasing
                        self.best_cost = cost_decrease
                        self.best_gains = test_gains.copy()
                        self.steps[i] *= self.step_scale
                        iteration_improved = True
                        print(" ✓ BETTER")
                    else:
                        # No improvement in either direction - shrink step
                        self.steps[i] *= self.step_shrink
                        print(" ✗ shrink step")

            # Update gains to best found
            self.gains = self.best_gains.copy()

            # Log iteration
            self.history.append({"iteration": iteration, "cost": self.best_cost, "gains": self.best_gains.copy()})

            # Check for early stopping
            if not iteration_improved:
                self.no_improvement_count += 1
                print(f"  [No improvement for {self.no_improvement_count} iterations]")
                if self.no_improvement_count >= self.max_no_improvement:
                    print(f"\n  [Early stopping: No improvement for {self.max_no_improvement} iterations]")
                    break
            else:
                self.no_improvement_count = 0


        print("\n" + "=" * 80)
        print("AUTO-TUNING COMPLETE")
        print("=" * 80)
        print(f"Best cost: {self.best_cost:.6f}")
        print(f"Best gains:")
        print(f"  Kp_roll={self.best_gains[0]:.4f}, Ki_roll={self.best_gains[1]:.4f}, Kd_roll={self.best_gains[2]:.4f}")
        print(f"  Kp_pitch={self.best_gains[3]:.4f}, Ki_pitch={self.best_gains[4]:.4f}, Kd_pitch={self.best_gains[5]:.4f}")
        print("=" * 80 + "\n")

        return self.best_gains, self.best_cost

    def save_history(self, filepath: str):
        """Save tuning history to CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "cost", "kp_roll", "ki_roll", "kd_roll", "kp_pitch", "ki_pitch", "kd_pitch"])
            for record in self.history:
                writer.writerow(
                    [
                        record["iteration"],
                        f"{record['cost']:.6f}",
                        f"{record['gains'][0]:.6f}",
                        f"{record['gains'][1]:.6f}",
                        f"{record['gains'][2]:.6f}",
                        f"{record['gains'][3]:.6f}",
                        f"{record['gains'][4]:.6f}",
                        f"{record['gains'][5]:.6f}",
                    ]
                )


def main():
    """Main function."""

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Short episodes for tuning
    env_cfg.episode_length_s = 10.0

    env = gym.make(args_cli.task, cfg=env_cfg)

    print("\n" + "=" * 80)
    print("EVOBOT V1 - PID AUTO-TUNING")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Episode length: {env_cfg.episode_length_s:.1f} seconds")
    print("=" * 80)

    # Get robot asset
    robot = env.unwrapped.scene["robot"]

    # Get timestep
    dt = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation

    # Initial gains
    init_gains = [
        args_cli.init_kp,
        args_cli.init_ki,
        args_cli.init_kd,
        args_cli.init_kp,
        args_cli.init_ki,
        args_cli.init_kd,
    ]

    # Create auto-tuner
    tuner = PIDAutoTuner(
        env=env,
        robot=robot,
        init_gains=init_gains,
        max_iterations=args_cli.max_iterations,
        tolerance=args_cli.tolerance,
        init_step=args_cli.init_step,
        step_scale=args_cli.step_scale,
        step_shrink=args_cli.step_shrink,
        eval_steps=args_cli.eval_steps,
        cost_weight_action=args_cli.cost_weight_action,
        dt=dt,
        device=args_cli.device,
    )

    # Run tuning
    best_gains, best_cost = tuner.tune()

    # Save results
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join("logs", f"pid_autotune_{timestamp}.csv")
    tuner.save_history(history_file)
    print(f"[INFO] Tuning history saved to: {history_file}")

    # Print command to test best gains
    print("\n" + "=" * 80)
    print("TEST BEST GAINS WITH:")
    print("=" * 80)
    print(f"./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/test_pid_balance_response.py \\")
    print(f"    --kp_roll {best_gains[0]:.4f} --ki_roll {best_gains[1]:.4f} --kd_roll {best_gains[2]:.4f} \\")
    print(f"    --kp_pitch {best_gains[3]:.4f} --ki_pitch {best_gains[4]:.4f} --kd_pitch {best_gains[5]:.4f}")
    print("=" * 80 + "\n")

    # Close environment
    env.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
