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
parser.add_argument("--seed", type=int, default=2, help="Seed for the environment for deterministic behavior.")

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
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

from isaaclab_assets import LeggedRobotV2EnvCfgTest

from isaaclab.managers import SceneEntityCfg


def run_simulator(env: ManagerBasedRLEnv):
    """Run the sensor test loop."""
    imu_cfg = SceneEntityCfg("imu")
    imu = env.scene[imu_cfg.name]
    
    # ray__caster_cfg = SceneEntityCfg("height_scanner")
    # ray__caster = env.scene[ray__caster_cfg.name]
    
    print("[INFO]: Resetting environment...")
    obs, _ = env.reset()
    count = 0

    num_joints = env.action_manager.action.shape[1]

    # Tạo tensor hành động cơ bản (0 cho tất cả)
    base_action = torch.zeros_like(env.action_manager.action)

    # Biên độ xoay (giá trị ±)
    amplitude = 1.0   # có thể tăng lên 0.5 hoặc 1.0 tùy scale
    duration = 0.5    # thời gian giữ mỗi khớp (giây)

    while simulation_app.is_running():
        # Random small torques to keep robot slightly moving
        actions = base_action

        # Step simulation
        obs, rewards, terminated, truncated, info = env.step(actions)
        count += 1
        
        print(actions)

        # Print every 100 steps
        if count % 100 == 0:
            print("=" * 80)
            print(f"[Step {count}] Sensor readings:")

            # ---- IMU ----
            imu_data = imu.data
            quat = imu.data.quat_w
            lin_acc = imu_data.lin_acc_b
            ang_vel = imu_data.ang_vel_b
            print(f"  Quaternion: {quat}")
            roll, pitch, yaw = euler_xyz_from_quat(quat)
            print(env.scene["robot"].data.root_state_w)  # robot base orientation
            print(f"  roll, pitch, yaw: {roll}, {pitch}, {yaw}")
            print(f"  IMU Linear Acc (m/s²): {lin_acc}")
            print(f"  IMU Angular Vel (rad/s): {ang_vel}")
            
            # raycaster_data = ray__caster.data
            # height = torch.max(raycaster_data.ray_hits_w[0])
            # print(f"  height (m): {height}")

            # # ---- Joint states ----
            # joint_pos = env.scene["robot"].data.joint_pos[0].cpu().numpy()
            # joint_vel = env.scene["robot"].data.joint_vel[0].cpu().numpy()
            # print(f"  Joint Pos (rad): {joint_pos}")
            # print(f"  Joint Vel (rad/s): {joint_vel}")

            # # ---- Contact forces ----
            # # if "contact_forces" in env.scene:
            # #     contact = env.scene["contact_forces"].data
            # #     total_force = contact.total_force_w[0].cpu().numpy()
            # #     print(f"  Contact Force (N): {total_force}")

        if count % 2000 == 0:
            print("[INFO]: Resetting environment for next test...")
            env.reset()

    env.close()
    print("[INFO]: Simulation ended.")



def main():
    """Main function."""
    # Configure environment
    env_cfg = LeggedRobotV2EnvCfgTest()
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