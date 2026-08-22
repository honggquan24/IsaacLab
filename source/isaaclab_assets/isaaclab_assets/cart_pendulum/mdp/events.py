# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event riêng của họ robot con lắc trên xe đẩy (đơn, kép, ba)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .rewards import resolve_joint_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_pendulum_chain(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
    hanging: bool = True,
    angle_noise: float = 0.1,
    velocity_noise: float = 0.05,
) -> None:
    """Đặt chuỗi con lắc về tư thế thõng xuống (swing-up) hoặc tư thế đứng.

    Mọi khớp quay bằng 0 tuyệt đối là cả chuỗi **thõng thẳng xuống**: khâu đầu bằng 0 là tư
    thế CAD, còn khâu sau bằng 0 nghĩa là thẳng hàng với khâu trước. Ngược lại, vị trí khớp
    mặc định của robot (``Revolute_1`` = π, các khâu sau = 0) là cả chuỗi **dựng đứng**.

    Vì vậy chỉ cần một cờ ``hanging`` là đổi được giữa bài swing-up và bài chỉ giữ thăng bằng,
    không phải viết hai event khác nhau.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(asset, asset_cfg)

    if hanging:
        target = torch.zeros((len(env_ids), len(joint_ids)), device=asset.device)
    else:
        target = asset.data.default_joint_pos[env_ids][:, joint_ids]

    joint_pos = target + torch.empty_like(target).uniform_(-angle_noise, angle_noise)
    joint_vel = torch.empty_like(target).uniform_(-velocity_noise, velocity_noise)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
