# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance-specific termination functions for Evobot V1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_when_fall(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt_angle: float = 1.0,  # ~57 degrees
) -> torch.Tensor:
    """Terminate when robot falls (excessive tilt angle).

    Args:
        env: Environment instance.
        asset_cfg: Robot asset configuration.
        max_tilt_angle: Maximum allowed tilt angle in radians before termination.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot = env.scene[asset_cfg.name]
    quat = robot.data.root_quat_w

    # Get roll and pitch from quaternion
    roll, pitch, _ = euler_xyz_from_quat(quat)

    # Check if tilt exceeds threshold
    tilt_magnitude = torch.sqrt(roll**2 + pitch**2)
    fallen = tilt_magnitude > max_tilt_angle

    return fallen
