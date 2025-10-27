# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for a legged robot.

.. code-block:: bash

    ./isaaclab.sh -p scripts/run_legged_robot_env.py --num_envs 32 --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the legged robot RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment for deterministic behavior.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from icecream import ic
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_assets import LeggedRobotV1EnvCfg


def run_simulator(env: ManagerBasedRLEnv):
    """Run the simulator loop.
    
    Args:
        env: The RL environment instance.
    """
    count = 0
    
    # Reset environment initially
    env.reset()
    print("[INFO]: Environment reset complete.")

    while simulation_app.is_running():
        # Reset environment periodically
        if count % 500 == 0 and count > 0:
            env.reset()
            print("-" * 80)
            print(f"[INFO]: Resetting environment at step {count}...")

        # Generate random joint efforts as actions
        actions = torch.ones_like(env.action_manager.action)
        # print(f"actions: {actions.shape}")
        # ic(actions)
        
        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Print information periodically
        if count % 100 == 0:
            print("-" * 80)
            print(f"[Step {count}]")
            print(f"  Observation shape: {obs['policy'].shape}")
            print(f"  Mean reward: {rewards.mean().item():.4f}")
            ic(obs)
            ic(rewards)
            ic(terminated)
            ic(truncated)
            ic(info)
            
            # # Print sensor information from the scene
            # scene = env.scene
            # if "height_scanner" in scene:
            #     max_height = torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item()
            #     print(f"  Max height scan value: {max_height:.4f}")
            
            # if "camera" in scene and args_cli.enable_cameras:
            #     rgb_shape = scene["camera"].data.output["rgb"].shape
            #     print(f"  Camera RGB shape: {rgb_shape}")

        # Update counter
        count += 1


def main():
    """Main function."""
    # Configure environment
    env_cfg = LeggedRobotV1EnvCfg()
    print(f"joint_pos_rel: {env_cfg.observations.policy.joint_pos_rel}")
    # print(env_cfg.scene.imu.history_length)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # Setup RL environment
    print("[INFO]: Creating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"[INFO]: Setup complete with {args_cli.num_envs} environment(s).")
    print(f"[INFO]: Observation space: {env.observation_manager.group_obs_dim}")
    print(f"[INFO]: Action space: {env.action_manager.action.shape}")

    # Run the simulator
    run_simulator(env)

    # Close environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO]: Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close sim app
        simulation_app.close()