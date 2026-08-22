# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env cfg của con lắc kép trên xe đẩy.

Kế thừa nguyên env cfg của con lắc đơn — scene, action, quan sát, reward và termination đều
tra khớp bằng ``Revolute_.*`` nên tự nhận thêm khâu thứ hai. Ở đây chỉ đổi robot, khung nhìn
và giới hạn ray.
"""

from isaaclab.utils import configclass

from isaaclab_assets.cart_pendulum.cart_pendulum_env_cfg import (
    CartPendulumEnvCfg,
    CartPendulumPositionEnvCfg,
)

from .cart_pendulum_double_cfg import CART_PENDULUM_DOUBLE_CFG, CART_PENDULUM_DOUBLE_RAIL_LIMIT


def use_double_pendulum(cfg: CartPendulumEnvCfg) -> None:
    """Thay robot con lắc đơn bằng con lắc kép và chỉnh khung nhìn cho vừa."""
    cfg.scene.robot = CART_PENDULUM_DOUBLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.terminations.cart_out_of_rail.params["limit"] = CART_PENDULUM_DOUBLE_RAIL_LIMIT - 0.05
    # ray ở z≈0.45, chuỗi dựng lên tới z≈0.89 nên phải lùi ra và nâng mắt nhìn
    cfg.viewer.eye = (3.2, 0.0, 1.3)
    cfg.viewer.lookat = (0.0, 0.0, 0.6)


@configclass
class CartPendulumDoubleEnvCfg(CartPendulumEnvCfg):
    """Swing-up con lắc kép: bắt đầu thõng xuống, lắc lên rồi giữ cả hai khâu thẳng đứng."""

    def __post_init__(self) -> None:
        super().__post_init__()
        use_double_pendulum(self)


@configclass
class CartPendulumDoublePositionEnvCfg(CartPendulumPositionEnvCfg):
    """Con lắc kép vừa giữ thăng bằng vừa bám mốc vị trí được lệnh."""

    def __post_init__(self) -> None:
        super().__post_init__()
        use_double_pendulum(self)
        # bám vị trí đã khó, không bắt swing-up cùng lúc: mọi env bắt đầu ở tư thế đứng
        self.events.reset_pendulum.params["hanging_prob"] = 0.0


@configclass
class CartPendulumDoublePositionPlayEnvCfg(CartPendulumDoublePositionEnvCfg):
    """Cấu hình dùng lúc quay video: ít env, episode dài để clip không bị reset giữa chừng."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
