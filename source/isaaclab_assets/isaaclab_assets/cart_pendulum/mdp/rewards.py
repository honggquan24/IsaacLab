# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TARGET JOINT
TARGET_JOINT = torch.tensor(
    [
        # index: joint_name                # comment
        0.0,
        2.0,
        2.0,
        0.0,
    ]
)


def cartpole_reward_joint_pos(
    env: ManagerBasedRLEnv, target: torch.Tensor = TARGET_JOINT, scale_pos: float = 5.0, scale_vel: float = 4.0
) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos  # shape: [batch, joints]
    joint_vel = robot.data.joint_vel

    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    # Move tensors to correct device
    device = joint_pos.device

    if target.device != device:
        target = target.to(device)

    err_pos = scale_pos * (joint_pos[:, 1] - target[0])
    err_vel = scale_vel * (joint_vel[:, 1] - target[1])

    total_err = err_pos + err_vel

    reward = torch.exp(-(total_err**2))

    return reward


def cartpole_reward_joint_vel(env: ManagerBasedRLEnv, target: torch.Tensor = TARGET_JOINT, scale: float = 0.8):
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel

    device = joint_vel.device
    if target.device != device:
        target = target.to(device)

    err_cart = torch.abs(joint_vel[:, 0]) - target[1]
    err_cart = torch.clamp(err_cart, min=0.0)

    err_pendulum = torch.abs(joint_vel[:, 1]) - target[2]
    err_pendulum = torch.clamp(err_pendulum, min=0.0)

    reward = torch.exp(-scale * (err_cart**2 + err_pendulum**2))
    return 1 - reward


def cartpole_reward_fall(
    env: ManagerBasedRLEnv,
    threshold_fall: float = 0.5,
    scale: float = 1.0,
):
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    theta = joint_pos[:, 1]

    err = torch.abs(theta) - threshold_fall
    err = torch.clamp(err, min=0.0)

    reward = 1.0 - torch.exp(-scale * err**2)
    return reward
