# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance-specific observation functions for Evobot V1."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obs_body_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get roll angle from IMU quaternion."""
    imu = env.scene[asset_cfg.name]
    quat = imu.data.quat_w
    roll, _, _ = euler_xyz_from_quat(quat)
    return roll.unsqueeze(-1)


def obs_body_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get pitch angle from IMU quaternion."""
    imu = env.scene[asset_cfg.name]
    quat = imu.data.quat_w
    _, pitch, _ = euler_xyz_from_quat(quat)
    return pitch.unsqueeze(-1)


def obs_body_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get yaw angle from IMU quaternion."""
    imu = env.scene[asset_cfg.name]
    quat = imu.data.quat_w
    _, _, yaw = euler_xyz_from_quat(quat)
    return yaw.unsqueeze(-1)


def lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get linear velocity in body frame from IMU."""
    imu = env.scene[asset_cfg.name]
    # IMU provides linear velocity in world frame, we return it directly
    # Note: For proper body frame velocity, rotation transformation would be needed
    return imu.data.lin_vel_b


def angl_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get angular velocity in body frame from IMU."""
    imu = env.scene[asset_cfg.name]
    return imu.data.ang_vel_b


def obs_pos_world(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get robot position in world frame relative to environment origin."""
    robot = env.scene[asset_cfg.name]
    # Get robot root position and subtract environment origin
    pos_w = robot.data.root_pos_w
    env_origins = env.scene.env_origins
    pos_rel = pos_w - env_origins
    return pos_rel
