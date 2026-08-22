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
    hanging_prob: float = 0.5,
    angle_noise: float = 0.1,
    velocity_noise: float = 0.05,
) -> None:
    """Đặt chuỗi con lắc về tư thế thõng xuống hoặc tư thế đứng, bốc ngẫu nhiên từng env.

    Mọi khớp quay bằng 0 tuyệt đối là cả chuỗi **thõng thẳng xuống**: khâu đầu bằng 0 là tư
    thế CAD, còn khâu sau bằng 0 nghĩa là thẳng hàng với khâu trước. Ngược lại, vị trí khớp
    mặc định của robot (``Revolute_1`` = π, các khâu sau = 0) là cả chuỗi **dựng đứng**.

    ``hanging_prob`` là xác suất mỗi env khởi động ở tư thế thõng: 1.0 là swing-up thuần,
    0.0 là chỉ giữ thăng bằng, 0.5 là trộn đôi.

    Trộn đôi thường học nhanh hơn hẳn swing-up thuần. Bài này có hai kỹ năng tách biệt — lắc
    lên và giữ — mà kỹ năng giữ chỉ tập được khi đã ở gần đỉnh. Nếu mọi env đều bắt đầu từ
    dưới thì policy phải tự lắc lên được trước đã, mà lắc lên bằng chính sách ngẫu nhiên thì
    hiếm, nên nó gần như không có mẫu nào ở vùng đỉnh để học giữ. Cho một nửa số env đứng sẵn
    thì hai kỹ năng được học song song, và phần thưởng ở vùng đỉnh cũng thành mục tiêu rõ ràng
    cho nửa còn lại nhắm tới.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_joint_ids(asset, asset_cfg)

    default = asset.data.default_joint_pos[env_ids][:, joint_ids]
    hanging = torch.rand(len(env_ids), 1, device=asset.device) < hanging_prob
    target = torch.where(hanging, torch.zeros_like(default), default)

    joint_pos = target + torch.empty_like(target).uniform_(-angle_noise, angle_noise)
    joint_vel = torch.empty_like(target).uniform_(-velocity_noise, velocity_noise)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
