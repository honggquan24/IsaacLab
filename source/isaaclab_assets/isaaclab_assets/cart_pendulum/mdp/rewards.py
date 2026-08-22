# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward riêng của họ robot con lắc trên xe đẩy (đơn, kép, ba).

Cùng dạng với cartpole gốc của Isaac Lab
(``isaaclab_tasks/manager_based/classic/cartpole/mdp/rewards.py``): phạt bình phương sai số
vị trí và phạt trị tuyệt đối vận tốc, không dùng exp. L2 có độ dốc ở mọi góc nên bài swing-up
vẫn có cái để bám, khác với ``exp(-e²/std²)`` bão hoà về 0 khi con lắc thõng.

Quy ước góc
-----------
``Revolute_1`` bằng 0 là khâu đầu **thõng xuống**; các khâu sau bằng 0 là **thẳng hàng với
khâu trước**. Nên mọi khớp quay bằng 0 tuyệt đối là cả chuỗi thõng thẳng, còn vị trí khớp mặc
định của robot (``Revolute_1`` = π, còn lại 0) là cả chuỗi dựng đứng. Vì vậy mặc định các hàm
ở đây lấy mục tiêu là ``default_joint_pos``: đổi tư thế đích chỉ cần sửa ``init_state``.

Khớp tra theo tên qua :class:`SceneEntityCfg` nên thêm khâu vào CAD không phải sửa gì ở đây.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def resolve_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    """Chỉ số các khớp mà ``asset_cfg`` chọn.

    Manager chỉ resolve những :class:`SceneEntityCfg` nằm trong ``params`` của term. Nếu term
    không truyền ``asset_cfg`` mà xài giá trị mặc định trong chữ ký hàm thì ``joint_ids`` vẫn
    còn là ``slice(None)`` và không lấy chỉ số ra được. Trường hợp đó tự tra theo tên rồi ghi
    ngược lại vào ``asset_cfg`` để lần sau khỏi tra.
    """
    if isinstance(asset_cfg.joint_ids, slice):
        if not asset_cfg.joint_names:
            raise ValueError(
                f"SceneEntityCfg cho '{asset_cfg.name}' không nêu joint_names nên không biết lấy khớp nào."
            )
        asset_cfg.joint_ids = asset.find_joints(asset_cfg.joint_names)[0]
    return asset_cfg.joint_ids


def joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, wrap: bool = False) -> torch.Tensor:
    """Lệch của các khớp so với vị trí mặc định. Shape là (num_envs, num_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = resolve_joint_ids(asset, asset_cfg)
    error = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    return wrap_to_pi(error) if wrap else error


def joint_pos_target_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wrap: bool = True,
) -> torch.Tensor:
    """Phạt bình phương lệch vị trí khớp so với vị trí mặc định, cộng trên các khớp.

    Giống ``joint_pos_target_l2`` của cartpole gốc, chỉ khác chỗ mục tiêu lấy từ
    ``default_joint_pos`` thay vì một hằng số: chuỗi nhiều khâu không có chung một góc đích.
    Đặt ``wrap=False`` cho khớp trượt, vì bọc về [-π, π] chỉ đúng với góc.
    """
    return torch.sum(torch.square(joint_deviation(env, asset_cfg, wrap=wrap)), dim=1)


def joint_pos_command_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "cart_position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
) -> torch.Tensor:
    """Phạt bình phương sai số giữa vị trí xe và mốc được lệnh."""
    asset: Articulation = env.scene[asset_cfg.name]
    cart_pos = asset.data.joint_pos[:, resolve_joint_ids(asset, asset_cfg)[0]]
    target = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(cart_pos - target)
