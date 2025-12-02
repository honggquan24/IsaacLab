# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import time

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_assets import LeggedRobotV2EnvCfgTest
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from icecream import ic

def main():
    """Main function."""
    # create environment configuration
    env_cfg = LeggedRobotV2EnvCfgTest()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene['robot']
    ray_caster = env.scene['height_scanner']
    contact_forces = env.scene['contact_forces']
    
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action) * 0.0 
            
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            
            # CẬP NHẬT height sau mỗi step
            # Lấy vị trí sensor
            # sensor_pos_z = ray_caster.data.pos_w[:, 2]  # [num_envs] - Vị trí Z của sensor

            # # Lấy tọa độ Z của điểm chạm
            # ray_hits_z = ray_caster.data.ray_hits_w[..., 2]  # [num_envs, num_rays]

            # # TÍNH KHOẢNG CÁCH (sensor height above ground)
            # # Cho environment đầu tiên:
            # valid_mask = ray_hits_z[0] > -1e6
            # if valid_mask.any():
            #     ground_z = torch.mean(ray_hits_z[0][valid_mask])  # Chiều cao trung bình của mặt đất
            #     robot_height = sensor_pos_z[0] - ground_z  # ← ĐÂY MỚI LÀ CHIỀU CAO ROBOT
                
            #     print(f"[Sensor Z position]: {sensor_pos_z[0].item():.4f}")  # VD: 0.3
            #     print(f"[Ground Z position]: {ground_z.item():.4f}")  # VD: 0.0
            #     print(f"[Robot Height]: {robot_height.item():.4f}")  # VD: 0.3
            print("contact sensor","-"*50, end="\t")
            ic(contact_forces)
            ic(contact_forces.data)
            ic(contact_forces.data.net_forces_w)
            
            # update counter
            count += 1
            time.sleep(0.5)
    
    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
