# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_when_fall(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w

    roll, _, _ = euler_xyz_from_quat(quat)

    upright = math.pi / 2
    threshold = math.pi / 180 * 50

    terminate = torch.abs(roll - upright) > threshold
    return terminate
