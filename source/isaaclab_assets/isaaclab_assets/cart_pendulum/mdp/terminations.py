# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Điều kiện kết thúc riêng của họ robot con lắc trên xe đẩy (đơn, kép, ba)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import joint_deviation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pendulum_fell(
    env: ManagerBasedRLEnv,
    limit_angle: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
) -> torch.Tensor:
    """Kết thúc khi có khâu nào lệch quá ``limit_angle`` rad so với tư thế mặc định.

    Dùng hàm này thay cho ``joint_pos_out_of_manual_limit`` của Isaac Lab vì khớp con lắc
    quay tự do không giới hạn: sau một vòng thì góc khớp thành 3π chứ không quay về π, so
    trực tiếp với một khoảng cố định sẽ sai. Ở đây lệch được wrap về [-π, π] trước khi so.
    """
    return torch.any(torch.abs(joint_deviation(env, asset_cfg, wrap=True)) > limit_angle, dim=1)


def cart_out_of_rail(
    env: ManagerBasedRLEnv,
    limit: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
) -> torch.Tensor:
    """Kết thúc khi xe chạy quá ``limit`` mét tính từ giữa ray.

    Giới hạn khớp trong USD là ±0.555 m; dừng sớm hơn một chút để policy học tránh đầu ray
    thay vì học cách tì vào đó.
    """
    return torch.any(torch.abs(joint_deviation(env, asset_cfg)) > limit, dim=1)
