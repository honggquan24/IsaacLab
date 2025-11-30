# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for a legged robot.

.. code-block:: bash

    ./isaaclab.sh -p scripts/run_legged_robot_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the legged robot RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic behavior.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

from isaaclab_assets import LeggedRobotV2EnvCfgTest

from isaaclab.managers import SceneEntityCfg


# FIX: Corrected TARGET_JOINT_POS angles
TARGET_JOINT_POS = torch.tensor([
    # index: joint_name                # comment
    0.0,                               # 0: Left_Revolute_01 (hip)
    0.0,                               # 1: Right_Revolute_01 (hip)
    math.radians(38.0),                # 2: Left_Revolute_02 (knee)
    math.radians(25.0),                # 3: Left_Revolute_03 (ankle)
    -math.radians(38.0),               # 4: Right_Revolute_02 (knee)
    -math.radians(25.0),               # 5: Right_Revolute_03 (ankle)
    
    math.radians(-13.0),               # 6: Left_Revolute_05 (passive) - FIXED
    math.radians(12.6),                # 7: Right_Revolute_05 (passive)
    math.radians(-13.0),               # 8: Left_Revolute_06 (passive) - FIXED
    math.radians(12.6),                # 9: Right_Revolute_06 (passive)
    
    0.0,                               # 10: Left_Revolute_04 (wheel)
    0.0,                               # 11: Right_Revolute_04 (wheel)
])


def run_simulator(env: ManagerBasedRLEnv):
    """Run the simulator and test joint position control."""
    
    print("\n" + "="*60)
    print("[INFO] Starting Robot Test")
    print("="*60 + "\n")
    
    # Get scene and robot
    scene = env.scene
    robot = scene['robot']
    
    print(f"[INFO] Robot Information:")
    print(f"  - Number of joints: {robot.num_joints}")
    print(f"  - Joint names: {robot.joint_names}")
    print(f"  - Number of environments: {env.num_envs}")
    print()
    
    # FIX: Reset environment FIRST before accessing observations
    observations, extras = env.reset()
    
    # Print observation shapes
    print(f"[INFO] Observation Shapes:")
    policy_obs = observations.get('policy')
    critic_obs = observations.get('critic')
    if policy_obs is not None:
        print(f"  - Policy obs shape: {policy_obs.shape}")
    if critic_obs is not None:
        print(f"  - Critic obs shape: {critic_obs.shape}")
    print()
    
    # Print initial rewards/terminations
    log = extras.get('log', {})
    print(f"[INFO] Initial Episode Info:")
    for key, value in log.items():
        if torch.is_tensor(value):
            # Handle both scalar and multi-dimensional tensors
            if value.dim() == 0:
                print(f"  - {key}: {value.item():.4f}")
            else:
                print(f"  - {key}: {value[0].item():.4f}")
    print()
    
    # Prepare target positions
    target_pos = TARGET_JOINT_POS.to(robot.device).unsqueeze(0)
    target_pos = target_pos.expand(env.num_envs, -1)
    
    print(f"[INFO] Target Joint Positions (first env):")
    for i, (name, angle) in enumerate(zip(robot.joint_names, target_pos[0])):
        print(f"  {i:2d}. {name:20s}: {angle.item():7.4f} rad ({math.degrees(angle.item()):7.2f}°)")
    print()
    
    # FIX: Use proper control loop with env.step()
    step_count = 0
    max_steps = 10000
    print_interval = 100
    
    print(f"[INFO] Running simulation for {max_steps} steps...")
    print()
    
    while simulation_app.is_running() and step_count < max_steps:
        
        # FIX: Use env.step() instead of manual simulation stepping
        # Generate zero actions (we're using position targets, not effort)
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=robot.device)
        
        # Step environment
        observations, rewards, terminated, truncated, extras = env.step(actions)
        
        # Print debug info periodically
        if step_count % print_interval == 0:
            current_joint_pos = robot.data.joint_pos
            error = torch.norm(current_joint_pos - target_pos, dim=-1)
            max_error = error.max().item()
            mean_error = error.mean().item()
            
            print(f"[Step {step_count:5d}] Error: max={max_error:.4f} rad, mean={mean_error:.4f} rad")
            
            # Print joint positions for first env
            if step_count % (print_interval * 5) == 0:
                print(f"\n[DEBUG] Current Joint Positions (env 0):")
                for i, (name, current, target) in enumerate(
                    zip(robot.joint_names, current_joint_pos[0], target_pos[0])
                ):
                    error_val = (current - target).abs().item()
                    print(f"  {i:2d}. {name:20s}: {current.item():7.4f} rad "
                          f"(target: {target.item():7.4f}, error: {error_val:.4f})")
                print()
            
            # Print rewards
            log = extras.get('log', {})
            reward_str = " | ".join([
                f"{key.split('/')[-1]}: {value.item():.3f}" if value.dim() == 0 
                else f"{key.split('/')[-1]}: {value[0].item():.3f}"
                for key, value in log.items()
                if 'Reward' in key and torch.is_tensor(value)
            ])
            if reward_str:
                print(f"         Rewards: {reward_str}")
            
            # Print terminations
            term_str = " | ".join([
                f"{key.split('/')[-1]}: {value.item():.0f}" if value.dim() == 0
                else f"{key.split('/')[-1]}: {value[0].item():.0f}"
                for key, value in log.items()
                if 'Termination' in key and torch.is_tensor(value)
            ])
            if term_str:
                print(f"         Terminations: {term_str}")
            
            print()
        
        # Check for resets
        if terminated.any() or truncated.any():
            reset_envs = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            print(f"\n[WARNING] Environments {reset_envs.tolist()} were reset at step {step_count}")
            print(f"  - Terminated: {terminated.sum().item()}")
            print(f"  - Truncated: {truncated.sum().item()}\n")
        
        step_count += 1
    
    print("\n" + "="*60)
    print("[INFO] Simulation Complete")
    print("="*60 + "\n")
    
    # Final statistics
    final_joint_pos = robot.data.joint_pos
    final_error = torch.norm(final_joint_pos - target_pos, dim=-1)
    print(f"[FINAL] Joint Position Error:")
    print(f"  - Max error: {final_error.max().item():.4f} rad ({math.degrees(final_error.max().item()):.2f}°)")
    print(f"  - Mean error: {final_error.mean().item():.4f} rad ({math.degrees(final_error.mean().item()):.2f}°)")
    print(f"  - Min error: {final_error.min().item():.4f} rad ({math.degrees(final_error.min().item()):.2f}°)")
    print()


def main():
    """Main function."""
    # Create environment
    env_cfg = LeggedRobotV2EnvCfgTest()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Set seed for reproducibility
    env.seed(args_cli.seed)
    
    # Run simulator
    run_simulator(env)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()