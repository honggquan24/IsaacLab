# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obs_body_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    imu = env.scene[asset_cfg.name]
    quat = imu.data.quat_w
    r, p, y = euler_xyz_from_quat(quat)
    return r.unsqueeze(-1)


def obs_body_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    imu = env.scene[asset_cfg.name]
    quat = imu.data.quat_w
    r, p, y = euler_xyz_from_quat(quat)
    return p.unsqueeze(-1)


def obs_body_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    imu = env.scene[asset_cfg.name]
    quat = imu.data.quat_w
    r, p, y = euler_xyz_from_quat(quat)
    return y.unsqueeze(-1)


def obs_pos_world(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w - env.scene.env_origins


def lin_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    imu = env.scene[asset_cfg.name]
    return imu.data.lin_vel_b[:, 1].unsqueeze(-1)


def angl_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    imu = env.scene[asset_cfg.name]
    return imu.data.ang_vel_b[:, 0].unsqueeze(-1)
