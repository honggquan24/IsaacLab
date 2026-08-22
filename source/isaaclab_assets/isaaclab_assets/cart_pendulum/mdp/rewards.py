# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward riêng của họ robot con lắc trên xe đẩy (đơn, kép, ba).

Cùng dạng với cartpole gốc của Isaac Lab
(``isaaclab_tasks/manager_based/classic/cartpole/mdp/rewards.py``): phạt bình phương sai số
vị trí và phạt trị tuyệt đối vận tốc, không dùng exp.

Reward chia làm hai pha, cắt nhau ở ``upright_angle``:

* chưa dựng lên → :func:`swing_up_height`, thưởng theo độ cao chuỗi;
* đã dựng lên → :func:`pendulum_is_upright` (thưởng cố định) cộng :func:`balance_pole_pos_l2`
  (L2 như cũ).

Hai pha loại trừ nhau nên mỗi lúc chỉ một cái chạy.

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


def pendulum_is_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
    upright_angle: float = 0.4,
) -> torch.Tensor:
    """1.0 khi MỌI khâu nằm trong ``upright_angle`` rad quanh tư thế đứng, 0.0 nếu không.

    Vừa là cổng chia pha cho các hàm dưới, vừa dùng thẳng làm reward thưởng cho việc lên
    được — xem chú thích ở :func:`swing_up_height` về lý do phải có phần thưởng đó.
    """
    within = torch.abs(joint_deviation(env, asset_cfg, wrap=True)) < upright_angle
    return torch.all(within, dim=1).float()


def swing_up_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
    upright_angle: float = 0.4,
) -> torch.Tensor:
    """Độ cao của chuỗi, chỉ tính khi CHƯA dựng lên. Bằng 1 lúc đứng, -1 lúc thõng.

    Đây là reward của pha swing-up. Dùng ``cos`` chứ không phải L2 vì L2 ở tư thế thõng cho
    -π² ≈ -9.9 mỗi bước: một hằng số phạt khổng lồ áp đảo mọi tín hiệu khác, trong khi ``cos``
    bị chặn trong [-1, 1] nên độ dốc của nó mới là thứ policy nhìn thấy.

    Lấy ``min`` trên các khâu chứ **không** phải ``mean``. Khâu sau đo góc so với khâu trước,
    nên với chuỗi ba khâu thõng thẳng xuống thì lệch là ``(π, 0, 0)`` và ``mean(cos)`` ra
    **+0.33** — bằng đúng điểm của tư thế "khâu 1 dựng lên, hai khâu sau quẹo ngang 90°".
    Tức là ``mean`` chấm tư thế tệ nhất ngang với tư thế nửa vời, và policy khai thác đúng chỗ
    đó: nó quay mạnh cho khâu đầu vẫy quanh đỉnh còn hai khâu sau văng lung tung, ăn điểm mà
    không bao giờ phải xếp thẳng chuỗi. ``min`` chấm theo khâu tệ nhất — thõng ra -1, nửa vời
    ra 0, thẳng đứng ra +1 — nên muốn điểm cao chỉ còn một cách là xếp thẳng cả chuỗi, đúng
    thứ mà cổng :func:`pendulum_is_upright` đòi.

    .. important::
        Pha "đã dựng" **phải** kèm một phần thưởng đủ lớn (:func:`pendulum_is_upright` với
        trọng số dương), nếu không sẽ có vực: ngay dưới ngưỡng, term này còn cho
        ``2·cos(0.4) ≈ 1.84``; vượt qua ngưỡng nó tắt và chỉ còn phạt L2 — tức là lắc lên
        được lại bị trừ điểm, và policy sẽ học cách lửng lơ ngay dưới ngưỡng mãi mãi.
    """
    height = torch.min(torch.cos(joint_deviation(env, asset_cfg, wrap=True)), dim=1).values
    return height * (1.0 - pendulum_is_upright(env, asset_cfg, upright_angle))


def balance_pole_pos_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
    upright_angle: float = 0.4,
) -> torch.Tensor:
    """L2 lệch góc như cũ, nhưng chỉ tính khi đã dựng lên — reward của pha giữ thăng bằng."""
    return joint_pos_target_l2(env, asset_cfg) * pendulum_is_upright(env, asset_cfg, upright_angle)


def joint_pos_command_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "cart_position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Slider_1"]),
    pole_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
    upright_angle: float = 0.4,
) -> torch.Tensor:
    """Phạt bình phương sai số giữa vị trí xe và mốc — chỉ khi chuỗi đã dựng lên.

    Bám mốc lúc con lắc còn thõng là mâu thuẫn trực tiếp với việc lắc lên: bơm năng lượng thì
    phải chạy qua chạy lại, còn mốc lại giữ xe đứng yên một chỗ. Cổng này tắt hẳn term lúc
    chưa lên, để policy lo swing-up trước rồi mới lo bám vị trí.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cart_pos = asset.data.joint_pos[:, resolve_joint_ids(asset, asset_cfg)[0]]
    target = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(cart_pos - target) * pendulum_is_upright(env, pole_cfg, upright_angle)
