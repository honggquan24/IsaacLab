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
from icecream import ic
import time
import isaaclab.sim as sim_utils 
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab_assets import LeggedRobotV1EnvCfg
from isaaclab.utils import configclass

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0 
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0 


            print("[INFO]: Resetting robot state...")
            scene.reset()

        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        # scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        # print("-------------------------------")
        # print(scene["camera"])
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        # print("-------------------------------")
        # print(scene["imu"])
        # print("Received imu of robot:")
        # print("Linear velocity: ", scene["imu"].data.lin_vel_b)
        # print("Angular velocity: ", scene["imu"].data.ang_vel_b)
        # print("Linear acceleration: ", scene["imu"].data.lin_acc_b)
        # print("Angular acceleration: ", scene["imu"].data.ang_acc_b)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Config Env
    env_cfg = LeggedRobotV1EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Config scene
    scene = InteractiveScene(env_cfg.scene)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene)

    # setup RL environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)
    # env.close()


if __name__ == "__main__":
    # run the main function
    try:
        main()
    # close sim app
    finally:
        simulation_app.close()
