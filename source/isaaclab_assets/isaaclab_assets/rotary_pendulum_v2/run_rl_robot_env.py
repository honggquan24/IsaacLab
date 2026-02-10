# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug script for Evobot V1 sensors.

Usage:
    ./isaaclab.sh -p ./source/isaaclab_assets/isaaclab_assets/rotary_pendulum_v2/run_rl_robot_env.py --device cpu
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Robot")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("is_running:", simulation_app.is_running())
# from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat, subtract_frame_transforms

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_assets.rotary_pendulum_v2.navigation.balance import RotaryPendulumV2BalanceEnvCfg

from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)


import torch
from icecream import ic
import time

def main():
    # Setup environment
    env_cfg = RotaryPendulumV2BalanceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    

    # Access sensors
    robot = env.scene['robot']
    # asset_cfg = SceneEntityCfg(name="robot", body_names="gripper_01", joint_names="left_gripper_joint")
    # asset_cfg.resolve(env.scene)
    # asset = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command("grip_ee_pose_left")
       
    # imu = env.scene['imu']
    # height_scanner = env.scene['height_scanner']
    # contact_left = env.scene['contact_forces_wheel_left']
    # contact_right = env.scene['contact_forces_wheel_right']

    # print("\n=== Evobot V1 Sensor Debug ===")
    # print(f"Sensors: IMU={imu is not None}, Height={height_scanner is not None}")
    # print(f"Contacts: Left={contact_left is not None}, Right={contact_right is not None}\n")

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 1500 == 0:
                env.reset()
            #     print(f"\n[Step {count}] Reset")
            # ic(asset_cfg.joint_ids)
            # ic(asset_cfg.body_ids)
            # ic(asset.data.body_pos_w)
            # ic(asset.data.joint_pos[:, asset_cfg.joint_ids])
            # gripper_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze(1)  # (num_envs, 3)
            # ic(gripper_pos_w)
            # ic(
            #     subtract_frame_transforms(
            #         asset.data.root_pos_w,   # base position in world
            #         asset.data.root_quat_w,  # base orientation in world
            #         gripper_pos_w,           # gripper position in world          
            #     )
            # )
            
            # Zero action
            action = torch.rand_like(env.action_manager.action)
            # print(action)
            obs, rew, terminated, truncated, info = env.step(action)

            # # 1. Projected gravity
            # proj_grav = robot.data.projected_gravity_b
            
            # # 2. Root orientation (quaternion -> euler)
            # quat = robot.data.root_quat_w
            # # Compute roll, pitch, yaw from quaternion
            # roll = torch.atan2(2*(quat[:, 0]*quat[:, 1] + quat[:, 2]*quat[:, 3]),
            #                 1 - 2*(quat[:, 1]**2 + quat[:, 2]**2))
            # pitch = torch.asin(2*(quat[:, 0]*quat[:, 2] - quat[:, 3]*quat[:, 1]))
            # yaw = torch.atan2(2*(quat[:, 0]*quat[:, 3] + quat[:, 1]*quat[:, 2]),
            #                 1 - 2*(quat[:, 2]**2 + quat[:, 3]**2))
            
            # # 3. Root height
            # height = robot.data.root_pos_w[:, 2]
            
            # # 4. Angle from bad_orientation calculation
            # angle = torch.acos(-proj_grav[:, 2]).abs()
            
            # if count % 5==0:
            #     print(
            #         f"roll: {torch.rad2deg(roll[0]).item():.2f}° | "
            #         f"pitch: {torch.rad2deg(pitch[0]).item():.2f}° | "
            #         f"yaw: {torch.rad2deg(yaw[0]).item():.2f}° | "
            #         f"height: {height[0].item():.3f}m | "
            #         f"angle: {torch.rad2deg(angle[0]).item():.2f}° |"
            #         f"proj_grav: {proj_grav[0].cpu().numpy()} | "

            #     )
                        
            # # Print when any env terminates
            # if terminated.any():
            #     for env_id in torch.where(terminated)[0]:
            #         print(f"\n[ENV {env_id}] TERMINATED at step {count}")
            #         print(f"  Projected Gravity: {proj_grav[env_id].cpu().numpy()}")
            #         print(f"  Roll:  {torch.rad2deg(roll[env_id]):.2f}°")
            #         print(f"  Pitch: {torch.rad2deg(pitch[env_id]):.2f}°")
            #         print(f"  Yaw:   {torch.rad2deg(yaw[env_id]):.2f}°")
            #         print(f"  Height: {height[env_id]:.3f}m")
            #         print(f"  Angle (bad_orientation): {torch.rad2deg(angle[env_id]):.2f}°")
            #         print(f"  Root position: {robot.data.root_pos_w[env_id].cpu().numpy()}")
            
            # # Print periodic summary for all envs
            # if count % 100 == 0:
            #     print(f"\n=== Step {count} ===")
            #     print(f"Roll range:  [{torch.rad2deg(roll.min()):.1f}°, {torch.rad2deg(roll.max()):.1f}°]")
            #     print(f"Pitch range: [{torch.rad2deg(pitch.min()):.1f}°, {torch.rad2deg(pitch.max()):.1f}°]")
            #     print(f"Height range: [{height.min():.3f}m, {height.max():.3f}m]")
        
            
            
            # # Print sensor data every 50 steps
            # if count % 50 == 0:
            #     print(f"\n[Step {count}]")
            #     print(f"Robot pos: {robot.data.root_pos_w[0, :3]}")

                # if imu:
                #     print(f"IMU quat: {imu.data.quat_w[0]}")
                #     print(f"IMU ang_vel: {imu.data.ang_vel_b[0]}")

                # if height_scanner:
                #     print(f"Height data: {height_scanner.data.ray_hits_w[0, :3, 2]}")

                # if contact_left:
                #     print(f"Contact L: {contact_left.data.net_forces_w[0].norm():.2f}")
                # if contact_right:
                #     print(f"Contact R: {contact_right.data.net_forces_w[0].norm():.2f}")

            count += 1
            # time.sleep(0.1)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
