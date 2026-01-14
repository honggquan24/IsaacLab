# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, euler_xyz_from_quat, wrap_to_pi, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_man(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def reward_wheel_speed(env, asset_name: str = "robot"):
    robot = env.scene[asset_name]

    # joint velocity: [num_envs, num_joints]
    joint_vel = robot.data.joint_vel

    # giả sử wheel nằm ở index 0, 1
    wheel_vel = joint_vel[:, 0:2]

    # khuyến khích |v|
    reward = torch.mean(torch.abs(wheel_vel), dim=1)

    return reward


def rpy_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    q_cmd_w = command[:, 3:7]
    q_body_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]

    q_err = quat_mul(q_cmd_w, quat_conjugate(q_body_w))
    r, p, y = euler_xyz_from_quat(q_err)

    return torch.sqrt(r**2 + p**2 + y**2)




def position_command_error_tanh_man(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error_man(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def rpy_alignment_imu(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    tolerance: float = 0.1,  # THÊM: tolerance zone (radians)
    scale: float = 3.0,  # FIX: Giảm từ 3.0 xuống 1.0
    axis_weights: tuple = (2.0, 2.0, 0.5),  # THÊM: ưu tiên roll/pitch
) -> torch.Tensor:
    """Reward for RPY alignment with tolerance zone."""
    imu = env.scene[imu_cfg.name]
    quat = imu.data.quat_w
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    if torch.isnan(quat).any() or torch.isinf(quat).any():
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    roll, pitch, yaw = euler_xyz_from_quat(quat)
    roll = torch.clamp(roll, -torch.pi, torch.pi)
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)
    yaw = torch.clamp(yaw, -torch.pi, torch.pi)

    target_roll, target_pitch, target_yaw = target_rpy

    roll_error = torch.abs(wrap_to_pi(roll - target_roll))
    pitch_error = torch.abs(wrap_to_pi(pitch - target_pitch))
    yaw_error = torch.abs(wrap_to_pi(yaw - target_yaw))

    # FIX: Apply tolerance
    roll_error = torch.clamp(roll_error - tolerance, min=0.0)
    pitch_error = torch.clamp(pitch_error - tolerance, min=0.0)
    yaw_error = torch.clamp(yaw_error - tolerance, min=0.0)

    # FIX MEMORY LEAK: Cache weights tensor instead of creating new one every call
    # Old: weights = torch.tensor(axis_weights, device=roll.device, dtype=roll.dtype)
    # This created 184k+ tensors/sec causing massive memory leak!
    w0, w1, w2 = axis_weights
    weighted_error = (
        w0 * torch.square(roll_error) +
        w1 * torch.square(pitch_error) +
        w2 * torch.square(yaw_error)
    ) / (w0 + w1 + w2)  # Normalize by total weight

    # FIX: Lower scale
    reward = torch.exp(-scale * weighted_error)
    reward = torch.clamp(reward, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)

    return reward