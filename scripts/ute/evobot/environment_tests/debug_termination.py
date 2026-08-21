# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "num_envs": 1})
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg("Isaac-Evobot-Velocity", num_envs=1)
    env = gym.make("Isaac-Evobot-Velocity", cfg=env_cfg)

    obs, _ = env.reset()

    print("\n=== MONITORING TERMINATION VALUES ===")
    for step in range(1000):
        # Random actions
        actions = torch.randn_like(torch.tensor(env.action_space.sample()))

        obs, reward, terminated, truncated, info = env.step(actions)

        # Debug termination values
        robot = env.scene["robot"]

        # 1. Projected gravity
        proj_grav = robot.data.projected_gravity_b

        # 2. Root orientation (quaternion -> euler)
        quat = robot.data.root_quat_w
        # Compute roll, pitch, yaw from quaternion
        roll = torch.atan2(
            2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        )
        pitch = torch.asin(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]))
        yaw = torch.atan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        )

        # 3. Root height
        height = robot.data.root_pos_w[:, 2]

        # 4. Angle from bad_orientation calculation
        angle = torch.acos(-proj_grav[:, 2]).abs()

        # Print when any env terminates
        if terminated.any():
            for env_id in torch.where(terminated)[0]:
                print(f"\n[ENV {env_id}] TERMINATED at step {step}")
                print(f"  Projected Gravity: {proj_grav[env_id].cpu().numpy()}")
                print(f"  Roll:  {torch.rad2deg(roll[env_id]):.2f}°")
                print(f"  Pitch: {torch.rad2deg(pitch[env_id]):.2f}°")
                print(f"  Yaw:   {torch.rad2deg(yaw[env_id]):.2f}°")
                print(f"  Height: {height[env_id]:.3f}m")
                print(f"  Angle (bad_orientation): {torch.rad2deg(angle[env_id]):.2f}°")
                print(f"  Root position: {robot.data.root_pos_w[env_id].cpu().numpy()}")

        # Print periodic summary for all envs
        if step % 100 == 0:
            print(f"\n=== Step {step} ===")
            print(f"Roll range:  [{torch.rad2deg(roll.min()):.1f}°, {torch.rad2deg(roll.max()):.1f}°]")
            print(f"Pitch range: [{torch.rad2deg(pitch.min()):.1f}°, {torch.rad2deg(pitch.max()):.1f}°]")
            print(f"Height range: [{height.min():.3f}m, {height.max():.3f}m]")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
