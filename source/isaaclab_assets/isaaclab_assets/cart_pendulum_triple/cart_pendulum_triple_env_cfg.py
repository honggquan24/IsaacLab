# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env cfg của con lắc ba trên xe đẩy.

Kế thừa nguyên env cfg của con lắc đơn — reward và termination tra khớp bằng ``Revolute_.*``
nên tự nhận đủ ba khâu. Ở đây chỉ đổi robot, khung nhìn và giới hạn ray.
"""

from isaaclab.utils import configclass

from isaaclab_assets.cart_pendulum.cart_pendulum_env_cfg import (
    CartPendulumEnvCfg,
    CartPendulumPositionEnvCfg,
)

from .cart_pendulum_triple_cfg import CART_PENDULUM_TRIPLE_CFG, CART_PENDULUM_TRIPLE_RAIL_LIMIT


def use_triple_pendulum(cfg: CartPendulumEnvCfg) -> None:
    """Thay robot con lắc đơn bằng con lắc ba và chỉnh khung nhìn cho vừa."""
    cfg.scene.robot = CART_PENDULUM_TRIPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.terminations.cart_out_of_rail.params["limit"] = CART_PENDULUM_TRIPLE_RAIL_LIMIT - 0.05
    # đo từ USD: ray ở z≈0.65, chuỗi dựng lên tới z≈1.31
    cfg.viewer.eye = (3.8, 0.0, 1.6)
    cfg.viewer.lookat = (0.0, 0.0, 0.8)
    cfg.scene.env_spacing = 3.0


@configclass
class CartPendulumTripleEnvCfg(CartPendulumEnvCfg):
    """Swing-up con lắc ba: bắt đầu thõng xuống, lắc lên rồi giữ cả ba khâu thẳng đứng."""

    def __post_init__(self) -> None:
        super().__post_init__()
        use_triple_pendulum(self)


@configclass
class CartPendulumTriplePositionEnvCfg(CartPendulumPositionEnvCfg):
    """Con lắc ba vừa giữ thăng bằng vừa bám mốc vị trí được lệnh."""

    def __post_init__(self) -> None:
        super().__post_init__()
        use_triple_pendulum(self)
        # bám vị trí đã khó, không bắt swing-up cùng lúc: mọi env bắt đầu ở tư thế đứng
        self.events.reset_pendulum.params["hanging_prob"] = 0.0


@configclass
class CartPendulumTriplePositionPlayEnvCfg(CartPendulumTriplePositionEnvCfg):
    """Cấu hình dùng lúc quay video: ít env, episode dài để clip không bị reset giữa chừng."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 4.0
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
