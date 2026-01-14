# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance-specific reward functions for Evobot V1."""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_force_balance(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize imbalance in applied joint torques between left and right joints."""
    robot: Articulation = env.scene[left_cfg.name]
    torque_left = robot.data.applied_torque[:, left_cfg.joint_ids]
    torque_right = robot.data.applied_torque[:, right_cfg.joint_ids]
    diff = torch.abs(torque_left) - torch.abs(torque_right)
    reward = torch.sum(torch.square(diff), dim=1)
    return reward


def rpy_alignment_imu(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    tolerance: float = 0.1,
    scale: float = 3.0,
    axis_weights: tuple = (2.0, 2.0, 0.5),
) -> torch.Tensor:
    """Reward for RPY alignment with tolerance zone.

    Designed for balance tasks where the robot should maintain upright orientation.
    """
    imu = env.scene[imu_cfg.name]
    quat = imu.data.quat_w
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    if torch.isnan(quat).any() or torch.isinf(quat).any():
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    roll, pitch, yaw = euler_xyz_from_quat(quat)
    roll = torch.clamp(roll, -math.pi, math.pi)
    pitch = torch.clamp(pitch, -math.pi, math.pi)
    yaw = torch.clamp(yaw, -math.pi, math.pi)

    target_roll, target_pitch, target_yaw = target_rpy

    roll_error = torch.abs(wrap_to_pi(roll - target_roll))
    pitch_error = torch.abs(wrap_to_pi(pitch - target_pitch))
    yaw_error = torch.abs(wrap_to_pi(yaw - target_yaw))

    roll_error = torch.clamp(roll_error - tolerance, min=0.0)
    pitch_error = torch.clamp(pitch_error - tolerance, min=0.0)
    yaw_error = torch.clamp(yaw_error - tolerance, min=0.0)

    w0, w1, w2 = axis_weights
    weighted_error = (
        w0 * torch.square(roll_error) +
        w1 * torch.square(pitch_error) +
        w2 * torch.square(yaw_error)
    ) / (w0 + w1 + w2)

    reward = torch.exp(-scale * weighted_error)
    reward = torch.clamp(reward, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)

    return reward


def height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.5,
    sigma: float = 0.1,
    min_height: float = 0.1,
) -> torch.Tensor:
    """Reward for maintaining base height close to target."""
    ray_caster = env.scene['height_scanner']
    sensor_pos_z = ray_caster.data.pos_w[:, 2]
    ray_hits_z = ray_caster.data.ray_hits_w[..., 2]

    valid_mask = ray_hits_z > -1e6
    masked_hits = ray_hits_z * valid_mask
    num_valid = valid_mask.sum(dim=-1).clamp(min=1)
    ground_z = masked_hits.sum(dim=-1) / num_valid

    robot_heights = torch.where(
        valid_mask.any(dim=-1),
        sensor_pos_z - ground_z,
        sensor_pos_z
    )

    height_error = robot_heights - target_height
    reward = torch.exp(-(height_error ** 2) / (2 * sigma ** 2))

    reward = torch.where(robot_heights < min_height, torch.zeros_like(reward), reward)

    reward = torch.clamp(reward, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)

    return reward


def angular_velocity_reward(
    env: ManagerBasedRLEnv,
    target_angular_vel: float = 0.0,
    scale: float = 5.0,
    max_vel: float = 20.0,
    axis_weights: tuple = (0.5, 0.5, 0.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for low angular velocity (encourages stable, upright position)."""
    robot = env.scene[asset_cfg.name]
    ang_vel = robot.data.root_ang_vel_w

    w0, w1, w2 = axis_weights
    weighted_vel = torch.stack([
        ang_vel[:, 0] * w0,
        ang_vel[:, 1] * w1,
        ang_vel[:, 2] * w2
    ], dim=-1)
    ang_vel_norm = torch.norm(weighted_vel, dim=-1)
    ang_vel_norm = torch.clamp(ang_vel_norm, 0.0, max_vel)

    ang_vel_error = torch.abs(ang_vel_norm - target_angular_vel)
    reward = torch.exp(-scale * ang_vel_error ** 2)

    reward = torch.clamp(reward, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)

    return reward


def linear_velocity_reward(
    env: ManagerBasedRLEnv,
    target_linear_vel: float = 0.0,
    scale: float = 5.0,
    max_vel: float = 10.0,
    axis_weights: tuple = (1.0, 1.0, 0.3),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining low linear velocity (encourages standing still)."""
    robot = env.scene[asset_cfg.name]
    lin_vel = robot.data.root_lin_vel_w

    w0, w1, w2 = axis_weights
    weighted_vel = torch.stack([
        lin_vel[:, 0] * w0,
        lin_vel[:, 1] * w1,
        lin_vel[:, 2] * w2
    ], dim=-1)

    lin_vel_norm = torch.norm(weighted_vel, dim=-1)
    lin_vel_norm = torch.clamp(lin_vel_norm, 0.0, max_vel)

    lin_vel_error = torch.abs(lin_vel_norm - target_linear_vel)
    reward_component = torch.exp(-scale * lin_vel_error ** 2)

    reward = torch.clamp(reward_component, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)

    return reward


def feet_contact_force_symmetry(
    env: ManagerBasedRLEnv,
    threshold_force: float = 30.0,
    max_ratio_diff: float = 0.7,
) -> torch.Tensor:
    """Reward for symmetric contact forces between left and right wheels."""
    fz_l = env.scene.sensors["contact_forces_wheel_left"].data.force_w[:, 2].clamp(min=0.0)
    fz_r = env.scene.sensors["contact_forces_wheel_right"].data.force_w[:, 2].clamp(min=0.0)

    total = fz_l + fz_r + 1e-6
    diff = torch.abs(fz_l - fz_r) / total

    symmetry = 1.0 - torch.clamp(diff / max_ratio_diff, 0.0, 1.0)

    contact_ok = torch.minimum(
        fz_l / threshold_force, fz_r / threshold_force
    ).clamp(0.0, 1.0)

    return symmetry * contact_ok
