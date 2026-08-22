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
    """Lệch của các khớp so với vị trí mặc định. Shape là (num_envs, num_joints).

    ``asset_cfg`` chọn được nhiều khớp, nên cùng một hàm dùng cho con lắc đơn, kép và ba: hễ
    ``joint_names`` là ``["Revolute_.*"]`` thì bao nhiêu khâu cũng vào hết.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = resolve_joint_ids(asset, asset_cfg)
    error = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    return wrap_to_pi(error) if wrap else error


def joint_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Vận tốc của các khớp ``asset_cfg`` chọn. Shape là (num_envs, num_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, resolve_joint_ids(asset, asset_cfg)]


"""
Giữ con lắc thăng bằng.
"""


def upright_pendulum_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
    std: float = 0.35,
) -> torch.Tensor:
    """Thưởng khi cả chuỗi con lắc thẳng và dựng đứng, dạng exp(-e²/std²).

    Với con lắc kép/ba, khâu đầu lệch so với tư thế đứng còn các khâu sau lệch so với khâu
    trước, vì vậy "thẳng đứng" đúng bằng "mọi khớp về vị trí mặc định". Lấy trung bình để
    trọng số không đổi theo số khâu.
    """
    return torch.mean(torch.exp(-torch.square(joint_deviation(env, asset_cfg, wrap=True) / std)), dim=1)


def pendulum_upright_cos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
) -> torch.Tensor:
    """Thưởng định hình cho bài swing-up: (1 + cos(lệch)) / 2, trung bình trên các khâu.

    Bằng 1 khi chuỗi dựng đứng và 0 khi thõng xuống, và quan trọng là **có độ dốc ở mọi góc**.
    :func:`upright_pendulum_exp` với std 0.35 ở tư thế thõng chỉ còn cỡ e⁻⁸⁰, tức phẳng lì,
    nên nếu chỉ có mình nó thì policy không có gì để bám mà lắc lên. Dùng cả hai: hàm này kéo
    con lắc đi lên, hàm exp lo phần đứng cho chính xác.
    """
    return torch.mean(0.5 * (1.0 + torch.cos(joint_deviation(env, asset_cfg, wrap=True))), dim=1)


def pendulum_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
) -> torch.Tensor:
    """Phạt bình phương vận tốc góc của mọi khâu để hạn chế rung."""
    return torch.sum(torch.square(joint_velocity(env, asset_cfg)), dim=1)


"""
Giữ xe đẩy quanh gốc ray.
"""


def cart_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
) -> torch.Tensor:
    """Phạt bình phương khoảng cách từ xe tới giữa ray."""
    return torch.sum(torch.square(joint_deviation(env, asset_cfg)), dim=1)


def cart_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
) -> torch.Tensor:
    """Phạt bình phương vận tốc xe."""
    return torch.sum(torch.square(joint_velocity(env, asset_cfg)), dim=1)


"""
Bám vị trí xe đẩy (các task ``-Position``).
"""


def track_cart_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "cart_position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
    std: float = 0.25,
) -> torch.Tensor:
    """Thưởng theo sai số vị trí xe so với lệnh, dạng exp(-e²/std²)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cart_pos = asset.data.joint_pos[:, resolve_joint_ids(asset, asset_cfg)[0]]
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
    index = resolve_joint_ids(asset, asset_cfg)[0]
    cart_pos = asset.data.joint_pos[:, index]
    cart_vel = asset.data.joint_vel[:, index]
    target = env.command_manager.get_command(command_name)[:, 0]
    closeness = torch.exp(-torch.square((cart_pos - target) / std))
    return closeness * torch.square(cart_vel)
