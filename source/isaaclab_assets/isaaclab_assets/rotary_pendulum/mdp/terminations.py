# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for Rotary Pendulum V2 (Furuta Pendulum)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_when_pivot_exceeds_limit(
    env: ManagerBasedRLEnv,
    pivot_joint_idx: int = 0,
    max_pivot_angle: float = math.pi / 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when pivot angle exceeds limit.

    This prevents the pivot arm from spinning continuously.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    theta1 = robot.data.joint_pos[:, pivot_joint_idx]
    return torch.abs(theta1) > max_pivot_angle
