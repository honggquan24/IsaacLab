# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward L2 cho bám vận tốc của evoBOT.

Isaac Lab core chỉ có biến thể ``*_exp``; hai hàm dưới đây trả về BÌNH PHƯƠNG sai
số (dùng kèm trọng số âm) nên phạt mạnh hơn khi lệch lớn. Trước đây được vá vào
``isaaclab.envs.mdp.rewards``; nay đặt tại package của dự án.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty-based reward for tracking yaw angular velocity command.

    Reward = - weight * (w_cmd - w)^2
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    ang_cmd = env.command_manager.get_command(command_name)[:, 2]
    ang_vel = asset.data.root_ang_vel_b[:, 2]

    error = ang_cmd - ang_vel
    cost = error**2

    return cost


def track_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty-based reward for tracking linear velocity commands (x, y).

    Reward = - weight * ||v_cmd_xy - v_xy||^2
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # command: [v_x, v_y, w_z]
    vel_cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_xy = asset.data.root_lin_vel_b[:, :2]

    error = vel_cmd_xy - vel_xy
    cost = torch.sum(error**2, dim=1)

    return cost
