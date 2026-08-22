# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward riêng của dự án con lắc đơn trên xe đẩy.

Quy ước góc
-----------
Trong USD xuất từ Onshape, ``Revolute_1`` bằng 0 là lúc con lắc **thõng xuống** — đó là tư
thế CAD. Tư thế đứng nằm ở góc :math:`\\pi`, và nó được đặt làm ``init_state.joint_pos`` của
:data:`CART_PENDULUM_CFG`. Vì vậy mọi hàm ở đây đo lệch so với ``default_joint_pos`` chứ
không so với 0: đổi tư thế mặc định thì reward tự đi theo, không phải sửa hằng số ở đây.

Khớp được tra theo tên qua :class:`SceneEntityCfg`, không dùng chỉ số cứng, nên thứ tự khớp
mà PhysX sinh ra có đổi cũng không sai.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def resolve_joint_index(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    """Chỉ số của khớp mà ``asset_cfg`` chọn.

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
    return asset_cfg.joint_ids[0]


def joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, wrap: bool = False) -> torch.Tensor:
    """Lệch của một khớp so với vị trí mặc định. Shape là (num_envs,)."""
    asset: Articulation = env.scene[asset_cfg.name]
    index = resolve_joint_index(asset, asset_cfg)
    error = asset.data.joint_pos[:, index] - asset.data.default_joint_pos[:, index]
    return wrap_to_pi(error) if wrap else error


"""
Giữ con lắc thăng bằng.
"""


def upright_pendulum_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_1"]),
    std: float = 0.35,
) -> torch.Tensor:
    """Thưởng khi con lắc gần tư thế đứng, dạng exp(-e²/std²)."""
    return torch.exp(-torch.square(joint_deviation(env, asset_cfg, wrap=True) / std))


def pendulum_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_1"]),
) -> torch.Tensor:
    """Phạt bình phương vận tốc góc con lắc để hạn chế rung."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.joint_vel[:, resolve_joint_index(asset, asset_cfg)])


"""
Giữ xe đẩy quanh gốc ray.
"""


def cart_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
) -> torch.Tensor:
    """Phạt bình phương khoảng cách từ xe tới giữa ray."""
    return torch.square(joint_deviation(env, asset_cfg))


def cart_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
) -> torch.Tensor:
    """Phạt bình phương vận tốc xe."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.joint_vel[:, resolve_joint_index(asset, asset_cfg)])


"""
Bám vị trí xe đẩy (task Isaac-Cart-Pendulum-Position).
"""


def track_cart_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "cart_position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
    std: float = 0.25,
) -> torch.Tensor:
    """Thưởng theo sai số vị trí xe so với lệnh, dạng exp(-e²/std²)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cart_pos = asset.data.joint_pos[:, resolve_joint_index(asset, asset_cfg)]
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
    cart_pos = asset.data.joint_pos[:, resolve_joint_index(asset, asset_cfg)]
    cart_vel = asset.data.joint_vel[:, resolve_joint_index(asset, asset_cfg)]
    target = env.command_manager.get_command(command_name)[:, 0]
    closeness = torch.exp(-torch.square((cart_pos - target) / std))
    return closeness * torch.square(cart_vel)
