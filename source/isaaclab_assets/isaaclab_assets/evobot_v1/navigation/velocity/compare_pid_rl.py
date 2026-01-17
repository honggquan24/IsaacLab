#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to compare PID controller vs RL policy for evobot velocity control.

This script evaluates both controllers on control smoothness metrics:
- Action rate (L2 norm of action differences)
- Action jerk (derivative of acceleration)
- Control effort (sum of squared actions)
- Velocity tracking error
- Success rate (episodes without termination)

Usage:
    # Compare with default PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/compare_pid_rl.py \
        --num_envs 16 --num_episodes 100

    # Compare with custom PID gains
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/compare_pid_rl.py \
        --num_envs 16 --num_episodes 100 \
        --kp_linear 2.0 --ki_linear 0.1 --kd_linear 0.05 \
        --kp_angular 2.0 --ki_angular 0.1 --kd_angular 0.05

    # Compare with trained RL model
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/velocity/compare_pid_rl.py \
        --num_envs 16 --num_episodes 100 \
        --rl_checkpoint logs/rsl_rl/evobot_velocity/model_500.pt
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_pickle, dump_yaml

# Task imports
from isaaclab_assets.evobot_v1.navigation.velocity import EvobotV1VelocityBalanceEnvCfg
from isaaclab_assets.evobot_v1.mdp import VelocityPIDActionTermCfg


class ControllerComparator:
    """Compare PID vs RL controllers on control smoothness metrics."""

    def __init__(
        self,
        env_cfg: EvobotV1VelocityBalanceEnvCfg,
        num_envs: int,
        num_episodes: int,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.device = device

        # Update environment config
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = device

        # Create environment
        self.env = ManagerBasedRLEnv(cfg=env_cfg)

        # Storage for metrics
        self.metrics = {
            "pid": self._init_metrics_dict(),
            "rl": self._init_metrics_dict(),
        }

    def _init_metrics_dict(self) -> dict:
        """Initialize metrics storage dictionary."""
        return {
            "action_rate": [],  # L2 norm of action differences
            "action_jerk": [],  # Derivative of action rate
            "control_effort": [],  # Sum of squared actions
            "lin_vel_error": [],  # Linear velocity tracking error
            "ang_vel_error": [],  # Angular velocity tracking error
            "episode_length": [],  # Steps per episode
            "success": [],  # Episode completed without termination
            "actions_history": [],  # Full action trajectories for frequency analysis
        }

    def evaluate_pid(
        self,
        kp_linear: float = 2.0,
        ki_linear: float = 0.1,
        kd_linear: float = 0.05,
        kp_angular: float = 2.0,
        ki_angular: float = 0.1,
        kd_angular: float = 0.05,
    ):
        """Evaluate PID controller.

        Args:
            kp_linear: Proportional gain for linear velocity
            ki_linear: Integral gain for linear velocity
            kd_linear: Derivative gain for linear velocity
            kp_angular: Proportional gain for angular velocity
            ki_angular: Integral gain for angular velocity
            kd_angular: Derivative gain for angular velocity
        """
        print(f"\n{'='*60}")
        print(f"Evaluating PID Controller")
        print(f"  Gains: Kp_lin={kp_linear}, Ki_lin={ki_linear}, Kd_lin={kd_linear}")
        print(f"         Kp_ang={kp_angular}, Ki_ang={ki_angular}, Kd_ang={kd_angular}")
        print(f"{'='*60}\n")

        # Create PID action term configuration
        pid_cfg = VelocityPIDActionTermCfg(
            asset_name="robot",
            kp_linear=kp_linear,
            ki_linear=ki_linear,
            kd_linear=kd_linear,
            kp_angular=kp_angular,
            ki_angular=ki_angular,
            kd_angular=kd_angular,
            wheel_base=0.2,  # Adjust based on evobot dimensions
            wheel_radius=0.05,  # Adjust based on evobot dimensions
            scale=[300.0, 300.0, 300.0, 100.0, 100.0],  # Match velocity_env_cfg scales
        )

        # Replace action manager with PID
        from isaaclab.managers import ActionManager
        from isaaclab.utils import configclass

        @configclass
        class PIDActionCfg:
            velocity_pid = pid_cfg

        # Create new action manager
        pid_action_manager = ActionManager(PIDActionCfg(), self.env)

        # Evaluation loop
        self._run_evaluation("pid", pid_action_manager)

    def evaluate_rl(self, checkpoint_path: str | None = None):
        """Evaluate RL policy.

        Args:
            checkpoint_path: Path to trained model checkpoint. If None, uses random policy.
        """
        print(f"\n{'='*60}")
        if checkpoint_path:
            print(f"Evaluating RL Policy from: {checkpoint_path}")
        else:
            print(f"Evaluating Random RL Policy (no checkpoint provided)")
        print(f"{'='*60}\n")

        # Load RL policy if checkpoint provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Try to load RSL-RL policy
            try:
                from rsl_rl.runners import OnPolicyRunner
                from isaaclab_assets.evobot_v1.navigation.velocity.agents import rsl_rl_ppo_cfg

                # Load policy
                policy = self._load_rsl_rl_policy(checkpoint_path, rsl_rl_ppo_cfg.EvobotVelocityPPORunnerCfg())
            except Exception as e:
                print(f"Warning: Could not load RL policy: {e}")
                print("Using random policy instead.")
                policy = None
        else:
            policy = None

        # Use environment's default action manager
        action_manager = self.env.action_manager

        # Evaluation loop
        self._run_evaluation("rl", action_manager, policy=policy)

    def _load_rsl_rl_policy(self, checkpoint_path: str, runner_cfg):
        """Load RSL-RL policy from checkpoint."""
        import torch.nn as nn

        # Create actor network
        actor_net = self._create_actor_network(
            input_dim=self.env.observation_manager.group_obs_dim["policy"][0],
            output_dim=self.env.action_manager.total_action_dim,
            hidden_dims=runner_cfg.policy.actor_hidden_dims,
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        actor_net.load_state_dict(checkpoint["model_state_dict"])
        actor_net.eval()

        return actor_net

    def _create_actor_network(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> torch.nn.Module:
        """Create actor network matching RSL-RL architecture."""
        import torch.nn as nn

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers).to(self.device)

    def _run_evaluation(self, controller_type: str, action_manager, policy=None):
        """Run evaluation for a controller.

        Args:
            controller_type: "pid" or "rl"
            action_manager: Action manager to use
            policy: RL policy network (if None, uses random actions)
        """
        # Reset environment
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]

        # Episode tracking
        episode_count = 0
        episodes_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Per-environment storage
        env_actions_history = [[] for _ in range(self.num_envs)]
        env_prev_action = torch.zeros(self.num_envs, action_manager.total_action_dim, device=self.device)
        env_prev_action_rate = torch.zeros(self.num_envs, action_manager.total_action_dim, device=self.device)
        env_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Evaluation loop
        step_count = 0
        max_steps = self.num_episodes * int(self.env.max_episode_length) // self.num_envs + 100

        while episode_count < self.num_episodes and step_count < max_steps:
            # Get action from policy or random
            with torch.no_grad():
                if policy is not None:
                    action = policy(obs)
                else:
                    # Random actions for baseline
                    action = 2.0 * torch.rand(self.num_envs, action_manager.total_action_dim, device=self.device) - 1.0

            # Process and apply actions
            action_manager.process_actions(action)
            action_manager.apply_actions()

            # Step environment
            obs_dict, rewards, terminated, truncated, info = self.env.step(action)
            obs = obs_dict["policy"]
            dones = terminated | truncated

            # Compute metrics for each environment
            processed_action = action_manager.get_term("velocity_pid" if controller_type == "pid" else "wheel_effort").processed_actions

            # Action rate (derivative of action)
            action_rate = processed_action - env_prev_action

            # Action jerk (derivative of action rate)
            action_jerk = action_rate - env_prev_action_rate

            # Control effort
            control_effort = torch.sum(processed_action**2, dim=-1)

            # Velocity tracking error
            robot = self.env.scene["robot"]
            vel_cmd = self.env.command_manager.get_command("base_velocity")
            vel_current = torch.stack([
                robot.data.root_lin_vel_b[:, 0],  # vx
                robot.data.root_ang_vel_b[:, 2],  # wz
            ], dim=-1)
            lin_vel_error = torch.abs(vel_cmd[:, 0] - vel_current[:, 0])
            ang_vel_error = torch.abs(vel_cmd[:, 1] - vel_current[:, 1])

            # Store actions for frequency analysis
            for env_id in range(self.num_envs):
                if not episodes_completed[env_id]:
                    env_actions_history[env_id].append(processed_action[env_id].cpu().numpy())

            # Update step count
            env_step_count += 1

            # Check for episode completion
            newly_done_mask = dones & ~episodes_completed

            if newly_done_mask.any():
                done_ids = torch.where(newly_done_mask)[0]

                for env_id in done_ids:
                    env_id_int = env_id.item()

                    # Store episode metrics
                    self.metrics[controller_type]["action_rate"].append(torch.mean(torch.norm(action_rate[env_id], dim=-1)).item())
                    self.metrics[controller_type]["action_jerk"].append(torch.mean(torch.norm(action_jerk[env_id], dim=-1)).item())
                    self.metrics[controller_type]["control_effort"].append(control_effort[env_id].item())
                    self.metrics[controller_type]["lin_vel_error"].append(lin_vel_error[env_id].item())
                    self.metrics[controller_type]["ang_vel_error"].append(ang_vel_error[env_id].item())
                    self.metrics[controller_type]["episode_length"].append(env_step_count[env_id].item())
                    self.metrics[controller_type]["success"].append(not terminated[env_id].item())

                    # Store action history
                    if len(env_actions_history[env_id_int]) > 0:
                        self.metrics[controller_type]["actions_history"].append(np.array(env_actions_history[env_id_int]))

                    # Mark as completed
                    episodes_completed[env_id] = True
                    episode_count += 1

                    # Reset tracking for this environment
                    env_actions_history[env_id_int] = []
                    env_step_count[env_id] = 0

                    # Progress
                    if episode_count % 10 == 0:
                        print(f"  Progress: {episode_count}/{self.num_episodes} episodes")

            # Update previous values
            env_prev_action = processed_action.clone()
            env_prev_action_rate = action_rate.clone()
            step_count += 1

        print(f"  Completed: {episode_count} episodes\n")

    def compare_and_save(self, output_dir: str = "comparison_results"):
        """Compare metrics and save results.

        Args:
            output_dir: Directory to save results
        """
        # Create output directory
        output_path = Path(output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Comparison Results")
        print(f"{'='*60}\n")

        # Compute statistics
        results = {}
        for controller in ["pid", "rl"]:
            results[controller] = {}
            for metric in ["action_rate", "action_jerk", "control_effort", "lin_vel_error", "ang_vel_error", "episode_length"]:
                data = self.metrics[controller][metric]
                if len(data) > 0:
                    results[controller][metric] = {
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                    }

            # Success rate
            success_data = self.metrics[controller]["success"]
            if len(success_data) > 0:
                results[controller]["success_rate"] = {
                    "value": float(np.mean(success_data)),
                    "count": f"{sum(success_data)}/{len(success_data)}",
                }

        # Print comparison
        print(f"{'Metric':<25} {'PID':>20} {'RL':>20}")
        print(f"{'-'*65}")

        for metric in ["action_rate", "action_jerk", "control_effort", "lin_vel_error", "ang_vel_error"]:
            if metric in results["pid"] and metric in results["rl"]:
                pid_val = f"{results['pid'][metric]['mean']:.4f} ± {results['pid'][metric]['std']:.4f}"
                rl_val = f"{results['rl'][metric]['mean']:.4f} ± {results['rl'][metric]['std']:.4f}"
                print(f"{metric:<25} {pid_val:>20} {rl_val:>20}")

        if "success_rate" in results["pid"] and "success_rate" in results["rl"]:
            pid_success = f"{results['pid']['success_rate']['value']*100:.1f}% ({results['pid']['success_rate']['count']})"
            rl_success = f"{results['rl']['success_rate']['value']*100:.1f}% ({results['rl']['success_rate']['count']})"
            print(f"{'success_rate':<25} {pid_success:>20} {rl_success:>20}")

        print(f"\n{'='*60}\n")

        # Save results
        dump_yaml(str(output_path / "results.yaml"), results)
        dump_pickle(str(output_path / "metrics.pkl"), self.metrics)

        # Plot comparisons
        self._plot_comparisons(output_path)

        print(f"Results saved to: {output_path}")

    def _plot_comparisons(self, output_path: Path):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("PID vs RL Control Comparison", fontsize=16)

        metrics_to_plot = [
            ("action_rate", "Action Rate (L2 norm)"),
            ("action_jerk", "Action Jerk"),
            ("control_effort", "Control Effort"),
            ("lin_vel_error", "Linear Velocity Error (m/s)"),
            ("ang_vel_error", "Angular Velocity Error (rad/s)"),
            ("episode_length", "Episode Length (steps)"),
        ]

        for idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]

            pid_data = self.metrics["pid"][metric]
            rl_data = self.metrics["rl"][metric]

            if len(pid_data) > 0 and len(rl_data) > 0:
                ax.boxplot([pid_data, rl_data], labels=["PID", "RL"])
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "comparison_plots.png", dpi=150)
        plt.close()

        # Frequency analysis (if enough data)
        self._plot_frequency_analysis(output_path)

    def _plot_frequency_analysis(self, output_path: Path):
        """Plot frequency spectrum of control signals."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Control Signal Frequency Analysis", fontsize=14)

        for idx, controller in enumerate(["pid", "rl"]):
            actions_history = self.metrics[controller]["actions_history"]

            if len(actions_history) > 0:
                # Take first episode's actions
                actions = actions_history[0]

                if len(actions) > 10:
                    # Compute FFT for wheel velocities
                    left_wheel = actions[:, 0]
                    freqs = np.fft.rfftfreq(len(left_wheel), d=self.env.step_dt)
                    fft_vals = np.abs(np.fft.rfft(left_wheel))

                    axes[idx].plot(freqs, fft_vals)
                    axes[idx].set_xlabel("Frequency (Hz)")
                    axes[idx].set_ylabel("Magnitude")
                    axes[idx].set_title(f"{controller.upper()} - Left Wheel")
                    axes[idx].grid(True, alpha=0.3)
                    axes[idx].set_xlim(0, 10)  # Focus on low frequencies

        plt.tight_layout()
        plt.savefig(output_path / "frequency_analysis.png", dpi=150)
        plt.close()

    def cleanup(self):
        """Clean up resources."""
        self.env.close()


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description="Compare PID vs RL for evobot velocity control")

    # Environment settings
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    # PID gains
    parser.add_argument("--kp_linear", type=float, default=2.0, help="PID Kp for linear velocity")
    parser.add_argument("--ki_linear", type=float, default=0.1, help="PID Ki for linear velocity")
    parser.add_argument("--kd_linear", type=float, default=0.05, help="PID Kd for linear velocity")
    parser.add_argument("--kp_angular", type=float, default=2.0, help="PID Kp for angular velocity")
    parser.add_argument("--ki_angular", type=float, default=0.1, help="PID Ki for angular velocity")
    parser.add_argument("--kd_angular", type=float, default=0.05, help="PID Kd for angular velocity")

    # RL checkpoint
    parser.add_argument("--rl_checkpoint", type=str, default=None, help="Path to trained RL checkpoint")

    # Output
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Output directory")

    args = parser.parse_args()

    # Load environment config
    env_cfg = EvobotV1VelocityBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Create comparator
    comparator = ControllerComparator(
        env_cfg=env_cfg,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=args.device,
    )

    try:
        # Evaluate PID
        comparator.evaluate_pid(
            kp_linear=args.kp_linear,
            ki_linear=args.ki_linear,
            kd_linear=args.kd_linear,
            kp_angular=args.kp_angular,
            ki_angular=args.ki_angular,
            kd_angular=args.kd_angular,
        )

        # Evaluate RL
        comparator.evaluate_rl(checkpoint_path=args.rl_checkpoint)

        # Compare and save
        comparator.compare_and_save(output_dir=args.output_dir)

    finally:
        comparator.cleanup()


if __name__ == "__main__":
    main()
