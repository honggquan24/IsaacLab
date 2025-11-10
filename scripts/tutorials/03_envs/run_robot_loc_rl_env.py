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
from isaaclab.scene import InteractiveScene

from isaaclab.envs.ui import ManagerBasedRLEnvWindow
import omni.ui as ui

import time

def run_simulator(env: ManagerBasedRLEnv):
    print(f"[NOTE: Extract information]")
    
    # Check type of scene
    print(type(env.scene))
    
    scene = env.scene
    robot = scene['robot']
    
    # Tạo cửa sổ UI cho environment
    env_window = ManagerBasedRLEnvWindow(env, window_name="My Debug HUD")
    
    # Thêm nội dung HUD vào layout chính của cửa sổ
    with env_window.ui_window_elements["main_vstack"]:
        ui.Label("=== ROBOT DEBUG HUD ===", style={"color": 0xffcc00ff, "font_size": 18})
        ui.Spacer(height=5)
        
        # Robot Info Section
        ui.Label("=== ROBOT INFO ===", style={"color": 0xff00ccff, "font_size": 16})
        text_num_joints = ui.Label("Number of joints: ---")
        text_joint_names = ui.Label("Joint names: ---", word_wrap=True)
        
        ui.Spacer(height=10)
        ui.Label("=== OBSERVATIONS ===", style={"color": 0xff00ccff, "font_size": 16})
        text_policy = ui.Label("Policy: ---", word_wrap=True)
        text_policy_shape = ui.Label("Policy shape: ---")
        text_critic = ui.Label("Critic: ---", word_wrap=True)
        text_critic_shape = ui.Label("Critic shape: ---")
        
        ui.Spacer(height=10)
        ui.Label("=== REWARDS (Initial) ===", style={"color": 0xff00ccff, "font_size": 16})
        text_alive_init = ui.Label("Alive: ---")
        text_terminating = ui.Label("Terminating: ---")
        text_rpy_init = ui.Label("RPY Alignment: ---")
        
        ui.Spacer(height=10)
        ui.Label("=== TERMINATIONS (Initial) ===", style={"color": 0xff00ccff, "font_size": 16})
        text_time_out = ui.Label("Time Out: ---")
        text_joint_vel_limit = ui.Label("Joint Vel Limit: ---")
        
        ui.Spacer(height=10)
        ui.Label("=== JOINT POSITIONS ===", style={"color": 0xff00ccff, "font_size": 16})
        text_joint_positions = ui.Label("Joint positions policy: ---", word_wrap=True)
        
        ui.Spacer(height=10)
        ui.Label("=== LIVE REWARDS ===", style={"color": 0xffcc00ff, "font_size": 16})
        text_alive_live = ui.Label("Alive (Live): ---")
        text_rpy_live = ui.Label("RPY Align (Live): ---")
    
    # Lấy thông tin robot và cập nhật UI
    print(f"Number of joints: {robot.num_joints}")
    print(f"Joint names: {robot.joint_names}")
    
    # Cập nhật thông tin robot vào UI
    text_num_joints.text = f"Number of joints: {robot.num_joints}"
    text_joint_names.text = f"Joint names: {', '.join(robot.joint_names)}"
    
    # Reset environment và in thông tin ban đầu
    observations, extras = env.reset()
    
    policy = observations['policy']
    print(f"policy: {policy}")
    print(f"len policy: {policy.shape}")
    
    # Cập nhật policy vào UI
    text_policy.text = f"Policy: {policy}"
    text_policy_shape.text = f"Policy shape: {policy.shape}"
    
    critic = observations['critic']
    print(f"critic: {critic}")
    print(f"len critic: {critic.shape}")
    
    # Cập nhật critic vào UI
    text_critic.text = f"Critic: {critic}"
    text_critic_shape.text = f"Critic shape: {critic.shape}"
    
    log = extras['log']
    alive = log['Episode_Reward/alive']
    print(f"alive: {alive}")
    text_alive_init.text = f"Alive: {alive}"

    terminating = log['Episode_Reward/terminating']
    print(f"terminating: {terminating}")
    text_terminating.text = f"Terminating: {terminating}"
    
    rpy_alignment = log['Episode_Reward/rpy_alignment']
    print(f"rpy_alignment: {rpy_alignment}")
    text_rpy_init.text = f"RPY Alignment: {rpy_alignment}"
    
    time_out = log['Episode_Termination/time_out']
    print(f"time_out: {time_out}")
    text_time_out.text = f"Time Out: {time_out}"
    
    joint_vel_limit = log['Episode_Termination/joint_vel_limit']
    print(f"joint_vel_limit: {joint_vel_limit}")
    text_joint_vel_limit.text = f"Joint Vel Limit: {joint_vel_limit}"
    
    # Joint positions policy
    joint_pos = policy[:][:,13:13+12]
    print(f"Joint positions policy {joint_pos}")
    text_joint_positions.text = f"Joint positions policy: {joint_pos}"
    
    while simulation_app.is_running():
                
        # IsaacLab simulation step
        action = torch.randn_like(env.action_manager.action)
        observations, rewards, terminated, truncated, extras = env.step(action)
            
        # Extract observation info
        policy = observations["policy"]
        critic = observations["critic"]

        # Extract reward logs (safe access)
        log = extras.get("log", {})
        alive = log.get("Episode_Reward/alive")
        rpy_alignment = log.get("Episode_Reward/rpy_alignment")
        time_out = log.get("Episode_Termination/time_out")
        joint_vel_limit = log.get("Episode_Termination/joint_vel_limit")
        
        # Cập nhật giá trị HUD live
        if alive is not None:
            text_alive_live.text = f"Alive (Live): {alive:.4f}"
        if rpy_alignment is not None:
            text_rpy_live.text = f"RPY Align (Live): {rpy_alignment:.4f}"
            
        # Cập nhật joint positions live
        joint_pos = policy[:][:,13:13+12]
        text_joint_positions.text = f"Joint positions policy: {joint_pos}"
        
        time.sleep(10)

    
def main():
    env_cfg = LeggedRobotV2EnvCfgTest()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    run_simulator(env)
    env.close()
    
if __name__ == "__main__":
    main()