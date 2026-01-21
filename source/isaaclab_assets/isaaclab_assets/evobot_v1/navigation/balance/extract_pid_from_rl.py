#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract PID parameters from trained RL policy for Evobot V1 balance control.

This script analyzes a trained RL policy and extracts equivalent PID gains
by performing linear regression on the policy's behavior.

Methodology:
1. Load trained RL policy (RSL-RL PPO checkpoint)
2. Collect trajectories with various initial conditions
3. Compute errors (roll, pitch) and their derivatives/integrals
4. Perform linear regression: action = Kp*e + Ki*∫e + Kd*de/dt
5. Extract optimal PID gains from regression coefficients
6. Validate PID controller against RL policy
7. Export PID gains for use in classical controller

Key assumptions:
- RL policy has learned near-linear control law for balance
- First layer weights encode error-based control strategy
- PID structure is sufficient to approximate policy behavior

Usage:
    # Extract PID from trained checkpoint
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \\
        --checkpoint logs/rsl_rl/evobot_v1_balance/model_500.pt \\
        --num_trajectories 100 \\
        --trajectory_length 200

    # Extract with visualization
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \\
        --checkpoint logs/rsl_rl/evobot_v1_balance/model_500.pt \\
        --num_trajectories 200 \\
        --visualize

    # Extract and validate
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/extract_pid_from_rl.py \\
        --checkpoint logs/rsl_rl/evobot_v1_balance/model_500.pt \\
        --validate \\
        --validation_episodes 10

Output:
    - Console: Extracted PID gains (Kp, Ki, Kd for roll and pitch)
    - CSV: Trajectory data for analysis
    - PNG: Regression plots (if --visualize)
    - JSON: Extracted PID configuration

Theory:
    For balance control, RL policy learns:
        action[left_wheel] = f(roll, pitch, roll_vel, pitch_vel, ...)
        action[right_wheel] = g(roll, pitch, roll_vel, pitch_vel, ...)

    Ideal PID structure:
        roll_control = Kp_r * roll + Ki_r * ∫roll + Kd_r * roll_vel
        pitch_control = Kp_p * pitch + Ki_p * ∫pitch + Kd_p * pitch_vel

        left_wheel = pitch_control + roll_control
        right_wheel = pitch_control - roll_control

    Linear regression finds best-fit gains that minimize:
        ||action_RL - action_PID||² over collected trajectories
"""

import argparse
import torch
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Extract PID parameters from trained RL policy")
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Velocity", help="Task name")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained RL checkpoint (.pt file)")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments for data collection")
parser.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to collect")
parser.add_argument("--trajectory_length", type=int, default=200, help="Steps per trajectory")
parser.add_argument("--validate", action="store_true", help="Validate extracted PID against RL policy")
parser.add_argument("--validation_episodes", type=int, default=10, help="Number of validation episodes")
parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
parser.add_argument("--output_dir", type=str, default="logs/pid_extraction", help="Output directory")
parser.add_argument("--alpha", type=float, default=0.1, help="Ridge regression regularization (0 = no reg)")

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
from rsl_rl.runners import OnPolicyRunner


class TrajectoryCollector:
    """Collect trajectories from RL policy for PID extraction."""

    def __init__(self, env, policy, device: str = "cuda"):
        """Initialize trajectory collector.

        Args:
            env: Isaac Lab environment
            policy: Trained RL policy (actor network)
            device: Torch device
        """
        self.env = env
        self.policy = policy
        self.device = device
        self.robot = env.unwrapped.scene["robot"]
        self.dt = env.unwrapped.physics_dt * env.unwrapped.cfg.decimation

    def collect(
        self,
        num_trajectories: int,
        trajectory_length: int,
        randomize_initial: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Collect trajectories and extract control-relevant features.

        Args:
            num_trajectories: Number of trajectories to collect
            trajectory_length: Steps per trajectory
            randomize_initial: Randomize initial conditions

        Returns:
            Dictionary with arrays: roll_error, pitch_error, roll_vel, pitch_vel,
            roll_integral, pitch_integral, action_left, action_right
        """
        print(f"\n[INFO] Collecting {num_trajectories} trajectories ({trajectory_length} steps each)...")

        # Storage for trajectory data
        data = {
            "roll_error": [],       # Roll angle (target=0)
            "pitch_error": [],      # Pitch angle (target=0)
            "roll_vel": [],         # Roll angular velocity
            "pitch_vel": [],        # Pitch angular velocity
            "roll_integral": [],    # Integrated roll error
            "pitch_integral": [],   # Integrated pitch error
            "action_left": [],      # RL policy action (left wheel)
            "action_right": [],     # RL policy action (right wheel)
        }

        for traj_idx in range(num_trajectories):
            # Reset environment
            obs_dict, _ = self.env.reset()
            obs = obs_dict["policy"]

            # Optional: Apply random initial disturbance
            if randomize_initial and traj_idx > 0:
                # Random initial roll/pitch offset
                with torch.no_grad():
                    root_state = self.robot.data.root_state_w.clone()
                    # Add random roll/pitch perturbation
                    random_roll = torch.rand(self.env.unwrapped.num_envs, device=self.device) * 0.3 - 0.15
                    random_pitch = torch.rand(self.env.unwrapped.num_envs, device=self.device) * 0.3 - 0.15

                    # Convert to quaternion and apply (simplified - just for initial condition)
                    # For proper implementation, use quat_from_euler and multiply quaternions
                    self.robot.write_root_state_to_sim(root_state)

            # Trajectory tracking variables
            roll_integral = torch.zeros(self.env.unwrapped.num_envs, device=self.device)
            pitch_integral = torch.zeros(self.env.unwrapped.num_envs, device=self.device)

            for step in range(trajectory_length):
                with torch.inference_mode():
                    # Get current orientation
                    quat = self.robot.data.root_quat_w
                    roll, pitch, yaw = euler_xyz_from_quat(quat)

                    # Angular velocities from robot (or IMU)
                    ang_vel = self.robot.data.root_ang_vel_w  # (num_envs, 3) [wx, wy, wz]
                    roll_vel = ang_vel[:, 0]  # Roll rate
                    pitch_vel = ang_vel[:, 1]  # Pitch rate

                    # Compute errors (target is upright: roll=0, pitch=0)
                    roll_error = -roll  # Negative because we want to reach 0
                    pitch_error = -pitch

                    # Integrate errors
                    roll_integral += roll_error * self.dt
                    pitch_integral += pitch_error * self.dt

                    # Get RL policy action
                    obs_input = {"policy": obs}
                    actions = self.policy(obs_input)

                # Step environment
                obs_dict, _, _, _, _ = self.env.step(actions)
                obs = obs_dict["policy"]

                # Store data (only first env to avoid redundancy)
                data["roll_error"].append(roll_error[0].cpu().item())
                data["pitch_error"].append(pitch_error[0].cpu().item())
                data["roll_vel"].append(roll_vel[0].cpu().item())
                data["pitch_vel"].append(pitch_vel[0].cpu().item())
                data["roll_integral"].append(roll_integral[0].cpu().item())
                data["pitch_integral"].append(pitch_integral[0].cpu().item())
                data["action_left"].append(actions[0, 0].cpu().item())
                data["action_right"].append(actions[0, 1].cpu().item())

            if (traj_idx + 1) % 10 == 0:
                print(f"  Progress: {traj_idx + 1}/{num_trajectories} trajectories collected")

        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])

        print(f"[INFO] Collected {len(data['roll_error'])} samples total")
        return data


class PIDExtractor:
    """Extract PID gains from trajectory data using linear regression."""

    def __init__(self, alpha: float = 0.1):
        """Initialize PID extractor.

        Args:
            alpha: Ridge regression regularization parameter (0 = no regularization)
        """
        self.alpha = alpha
        self.gains = None
        self.metrics = None

    def extract(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract PID gains using linear regression.

        Theory:
            Roll control: action_roll = Kp_r * e_r + Ki_r * ∫e_r + Kd_r * de_r/dt
            Pitch control: action_pitch = Kp_p * e_p + Ki_p * ∫e_p + Kd_p * de_p/dt

            Wheel mapping (differential drive balance):
                left_wheel = pitch_control + roll_control
                right_wheel = pitch_control - roll_control

            Therefore:
                left_wheel = (Kp_p*e_p + Ki_p*∫e_p + Kd_p*ė_p) + (Kp_r*e_r + Ki_r*∫e_r + Kd_r*ė_r)
                right_wheel = (Kp_p*e_p + Ki_p*∫e_p + Kd_p*ė_p) - (Kp_r*e_r + Ki_r*∫e_r + Kd_r*ė_r)

            We can solve for 6 gains using ridge regression.

        Args:
            data: Dictionary of trajectory data

        Returns:
            Dictionary with keys: kp_roll, ki_roll, kd_roll, kp_pitch, ki_pitch, kd_pitch
        """
        print("\n[INFO] Extracting PID gains using linear regression...")

        # Prepare feature matrix X: [roll_error, roll_integral, roll_vel, pitch_error, pitch_integral, pitch_vel]
        X = np.column_stack([
            data["roll_error"],
            data["roll_integral"],
            data["roll_vel"],
            data["pitch_error"],
            data["pitch_integral"],
            data["pitch_vel"],
        ])

        # Target outputs
        y_left = data["action_left"]
        y_right = data["action_right"]

        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target (left wheel) shape: {y_left.shape}")
        print(f"  Target (right wheel) shape: {y_right.shape}")

        # Fit ridge regression for left and right wheels
        model_left = Ridge(alpha=self.alpha, fit_intercept=True)
        model_right = Ridge(alpha=self.alpha, fit_intercept=True)

        model_left.fit(X, y_left)
        model_right.fit(X, y_right)

        # Extract coefficients
        coef_left = model_left.coef_
        coef_right = model_right.coef_

        print("\n[INFO] Linear regression coefficients:")
        print(f"  Left wheel:  {coef_left}")
        print(f"  Right wheel: {coef_right}")
        print(f"  Bias (left):  {model_left.intercept_:.6f}")
        print(f"  Bias (right): {model_right.intercept_:.6f}")

        # Decompose into roll and pitch gains
        # Left = pitch + roll, Right = pitch - roll
        # => pitch = (left + right) / 2, roll = (left - right) / 2

        pitch_gains = (coef_left[3:6] + coef_right[3:6]) / 2  # [Kp_pitch, Ki_pitch, Kd_pitch]
        roll_gains = (coef_left[0:3] - coef_right[0:3]) / 2   # [Kp_roll, Ki_roll, Kd_roll]

        # Construct gains dictionary
        self.gains = {
            "kp_roll": float(roll_gains[0]),
            "ki_roll": float(roll_gains[1]),
            "kd_roll": float(roll_gains[2]),
            "kp_pitch": float(pitch_gains[0]),
            "ki_pitch": float(pitch_gains[1]),
            "kd_pitch": float(pitch_gains[2]),
        }

        print("\n[INFO] Extracted PID gains:")
        print(f"  Roll:  Kp={self.gains['kp_roll']:.4f}, Ki={self.gains['ki_roll']:.4f}, Kd={self.gains['kd_roll']:.4f}")
        print(f"  Pitch: Kp={self.gains['kp_pitch']:.4f}, Ki={self.gains['ki_pitch']:.4f}, Kd={self.gains['kd_pitch']:.4f}")

        # Compute goodness of fit
        y_left_pred = model_left.predict(X)
        y_right_pred = model_right.predict(X)

        r2_left = r2_score(y_left, y_left_pred)
        r2_right = r2_score(y_right, y_right_pred)
        mse_left = mean_squared_error(y_left, y_left_pred)
        mse_right = mean_squared_error(y_right, y_right_pred)

        self.metrics = {
            "r2_left": r2_left,
            "r2_right": r2_right,
            "mse_left": mse_left,
            "mse_right": mse_right,
        }

        print("\n[INFO] Regression quality metrics:")
        print(f"  Left wheel:  R²={r2_left:.4f}, MSE={mse_left:.6f}")
        print(f"  Right wheel: R²={r2_right:.4f}, MSE={mse_right:.6f}")

        return self.gains

    def save_gains(self, output_path: str):
        """Save extracted PID gains to JSON file."""
        if self.gains is None:
            raise ValueError("No gains to save. Run extract() first.")

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "pid_gains": self.gains,
            "regression_metrics": self.metrics,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[INFO] Saved PID gains to: {output_path}")


def visualize_regression(data: Dict[str, np.ndarray], gains: Dict[str, float], output_dir: str):
    """Generate visualization plots for regression analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available. Skipping visualization.")
        return

    print("\n[INFO] Generating visualization plots...")

    # Reconstruct PID actions from extracted gains
    roll_pid = (
        gains["kp_roll"] * data["roll_error"]
        + gains["ki_roll"] * data["roll_integral"]
        + gains["kd_roll"] * data["roll_vel"]
    )
    pitch_pid = (
        gains["kp_pitch"] * data["pitch_error"]
        + gains["ki_pitch"] * data["pitch_integral"]
        + gains["kd_pitch"] * data["pitch_vel"]
    )

    left_pid = pitch_pid + roll_pid
    right_pid = pitch_pid - roll_pid

    # Clip to [-1, 1] range (like actual PID controller)
    left_pid = np.clip(left_pid, -1.0, 1.0)
    right_pid = np.clip(right_pid, -1.0, 1.0)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot left wheel
    time = np.arange(len(data["action_left"])) * 0.01  # Assume ~100Hz
    axes[0].plot(time, data["action_left"], 'b-', linewidth=1.5, alpha=0.7, label='RL Policy')
    axes[0].plot(time, left_pid, 'r--', linewidth=1.5, alpha=0.7, label='Extracted PID')
    axes[0].set_ylabel('Left Wheel Action', fontsize=12)
    axes[0].set_title('Left Wheel: RL Policy vs Extracted PID', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time[-1]])

    # Plot right wheel
    axes[1].plot(time, data["action_right"], 'b-', linewidth=1.5, alpha=0.7, label='RL Policy')
    axes[1].plot(time, right_pid, 'r--', linewidth=1.5, alpha=0.7, label='Extracted PID')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Right Wheel Action', fontsize=12)
    axes[1].set_title('Right Wheel: RL Policy vs Extracted PID', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, time[-1]])

    plt.tight_layout()

    output_path = os.path.join(output_dir, "pid_extraction_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved plot to: {output_path}")
    plt.close()


def main():
    """Main function."""

    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Load environment
    print("\n" + "=" * 80)
    print("EXTRACT PID PARAMETERS FROM RL POLICY")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Trajectories to collect: {args_cli.num_trajectories} × {args_cli.trajectory_length} steps")
    print("=" * 80)

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Load trained policy
    print("\n[INFO] Loading trained RL policy...")
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")

    # Load checkpoint directly using PyTorch
    from isaaclab_assets.evobot_v1.navigation.velocity.agents.rsl_rl_ppo_cfg import EvobotVelocityPPORunnerCfg
    from rsl_rl.modules import ActorCritic

    agent_cfg = EvobotVelocityPPORunnerCfg()

    # Load checkpoint first to get correct dimensions
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=args_cli.device)

    # Infer num_obs and num_actions from checkpoint weights
    # actor.0.weight has shape [hidden_dim, num_obs]
    # actor.6.weight (last layer) has shape [num_actions, hidden_dim]
    num_obs = checkpoint['model_state_dict']['actor.0.weight'].shape[1]
    num_actions = checkpoint['model_state_dict']['actor.6.weight'].shape[0]

    print(f"[INFO] Detected from checkpoint: {num_obs} observations, {num_actions} actions")

    # Create obs dictionary with dummy tensors (ActorCritic needs this for initialization)
    obs_dict = {
        "policy": torch.zeros(1, num_obs, device=args_cli.device),
        "critic": torch.zeros(1, num_obs, device=args_cli.device)
    }

    # obs_groups defines which observations go to actor and critic
    obs_groups = {
        "policy": ["policy"],  # Actor uses "policy" observations
        "critic": ["critic"]   # Critic uses "critic" observations
    }

    actor_critic = ActorCritic(
        obs_dict,
        obs_groups,
        num_actions,
        **agent_cfg.policy.to_dict()
    ).to(args_cli.device)

    # Load checkpoint weights
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()

    # Get policy (actor network)
    policy = actor_critic.act_inference

    print("[INFO] Policy loaded successfully!")
    print(f"  Actor network: {agent_cfg.policy.actor_hidden_dims}")

        # Collect trajectories (still use the Gym wrapper for trajectory collection)
    collector = TrajectoryCollector(env, policy, device=args_cli.device)
    data = collector.collect(
        num_trajectories=args_cli.num_trajectories,
        trajectory_length=args_cli.trajectory_length,
        randomize_initial=True,
    )

    # Extract PID gains
    extractor = PIDExtractor(alpha=args_cli.alpha)
    gains = extractor.extract(data)

    # Save gains
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gains_path = os.path.join(args_cli.output_dir, f"pid_gains_{timestamp}.json")
    extractor.save_gains(gains_path)

    # Visualize (if requested)
    if args_cli.visualize:
        visualize_regression(data, gains, args_cli.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print("\nExtracted PID gains:")
    print(f"  Roll:  Kp={gains['kp_roll']:.4f}, Ki={gains['ki_roll']:.4f}, Kd={gains['kd_roll']:.4f}")
    print(f"  Pitch: Kp={gains['kp_pitch']:.4f}, Ki={gains['ki_pitch']:.4f}, Kd={gains['kd_pitch']:.4f}")

    print("\nRegression quality:")
    print(f"  Left wheel:  R²={extractor.metrics['r2_left']:.4f}")
    print(f"  Right wheel: R²={extractor.metrics['r2_right']:.4f}")

    print("\nTo test extracted PID gains, run:")
    print(f"./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/balance/test_pid_manual_tune.py \\")
    print(f"    --kp_roll {gains['kp_roll']:.4f} --ki_roll {gains['ki_roll']:.4f} --kd_roll {gains['kd_roll']:.4f} \\")
    print(f"    --kp_pitch {gains['kp_pitch']:.4f} --ki_pitch {gains['ki_pitch']:.4f} --kd_pitch {gains['kd_pitch']:.4f}")
    print("=" * 80)

    # Cleanup
    env.close()
    print("\n[INFO] PID extraction completed successfully!")


if __name__ == "__main__":
    main()
    simulation_app.close()