#!/usr/bin/env python3
"""Quick script to check robot's natural pitch angle when upright"""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Evobot-V1-Velocity-Play")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat
import numpy as np

# Create environment
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=env_cfg)
env.reset()

# Get IMU
imu = env.unwrapped.scene["imu"]

# Get pitch
quat = imu.data.quat_w
quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
roll, pitch, yaw = euler_xyz_from_quat(quat)

print("\n" + "="*60)
print("ROBOT NATURAL ORIENTATION (when upright)")
print("="*60)
print(f"Roll:  {roll[0].item():+.4f} rad ({np.rad2deg(roll[0].item()):+7.2f}°)")
print(f"Pitch: {pitch[0].item():+.4f} rad ({np.rad2deg(pitch[0].item()):+7.2f}°)")
print(f"Yaw:   {yaw[0].item():+.4f} rad ({np.rad2deg(yaw[0].item()):+7.2f}°)")
print("="*60)
print("\nIf pitch is large (e.g., 1.57 rad ≈ 90°), this means:")
print("- Robot body is designed with a pitch offset")
print("- OR IMU is mounted at an angle")
print("- The RL pitch tracking script now handles this by using RELATIVE pitch")
print("="*60 + "\n")

env.close()
simulation_app.close()
