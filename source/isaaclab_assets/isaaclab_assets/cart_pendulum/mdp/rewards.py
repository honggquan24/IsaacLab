# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

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


"""
Bám vị trí xe đẩy (task Isaac-Cart-Pendulum-Position).

Nhóm hàm dưới đây tra khớp qua :class:`SceneEntityCfg` thay vì chỉ số cứng, nên đổi thứ tự
khớp trong USD cũng không sai.
"""


def track_cart_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "cart_position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
    std: float = 0.25,
) -> torch.Tensor:
    """Thưởng theo sai số vị trí xe so với lệnh, dạng exp(-e²/std²)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cart_pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    target = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square((cart_pos - target) / std))


def cart_velocity_near_goal_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "cart_position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
    std: float = 0.25,
) -> torch.Tensor:
    """Phạt vận tốc xe, có trọng số theo mức độ gần mục tiêu.

    Ở xa mục tiêu thì gần như không phạt (xe được phép chạy nhanh), tới nơi mới phạt mạnh,
    nên xe dừng hẳn tại mốc thay vì dao động quanh nó.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cart_pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    cart_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[0]]
    target = env.command_manager.get_command(command_name)[:, 0]
    closeness = torch.exp(-torch.square((cart_pos - target) / std))
    return closeness * torch.square(cart_vel)


def upright_pendulum_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_1"]),
    std: float = 0.35,
) -> torch.Tensor:
    """Thưởng khi con lắc dựng đứng (góc khớp về 0)."""
    asset: Articulation = env.scene[asset_cfg.name]
    theta = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids[0]])
    return torch.exp(-torch.square(theta / std))


def pendulum_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_1"]),
) -> torch.Tensor:
    """Phạt bình phương vận tốc góc con lắc để hạn chế rung."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids[0]])
