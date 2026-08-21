# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for Rotary Pendulum V2 (Furuta Pendulum).

Observations for the pendulum swing-up/balance task:
- Joint positions: pivot angle (theta1), pendulum angle (theta2)
- Joint velocities: pivot angular velocity, pendulum angular velocity
- Trigonometric observations: sin/cos of angles for smooth representation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obs_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get joint positions [theta1, theta2]."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos


def obs_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get joint velocities [dtheta1, dtheta2]."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel


def obs_joint_pos_sin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get sin of joint positions [sin(theta1), sin(theta2)].

    Trigonometric representation avoids discontinuity at +-pi.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sin(robot.data.joint_pos)


def obs_joint_pos_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get cos of joint positions [cos(theta1), cos(theta2)].

    Trigonometric representation avoids discontinuity at +-pi.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.cos(robot.data.joint_pos)
