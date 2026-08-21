# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward term điều hướng của evoBOT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
) -> torch.Tensor:
    """Reward for tracking position command using tanh."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return 1.0 - torch.tanh(distance / std)


def heading_command_error_abs(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Penalty for heading error (absolute value)."""
    command = env.command_manager.get_command(command_name)
    heading = command[:, 3]
    return torch.abs(heading)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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


def position_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Discrete bonus when robot reaches target position."""
    command = env.command_manager.get_command(command_name)
    des_pos = command[:, :2]
    distance = torch.norm(des_pos, dim=1)
    return (distance < threshold).float()


def navigation_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for moving towards goal."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]

    vel_b = env.scene["robot"].data.root_lin_vel_b[:, :2]
    direction = torch.nn.functional.normalize(des_pos_b, dim=1, eps=1e-6)

    vel_towards_target = torch.sum(vel_b * direction, dim=1)
    return scale * torch.clamp(vel_towards_target, min=0.0)


def forward_velocity_tracking(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Reward for forward velocity."""
    return env.scene["robot"].data.root_lin_vel_b[:, 0]


def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for lateral velocity (y-axis in body frame)."""
    lat_vel = env.scene["robot"].data.root_lin_vel_b[:, 1]
    return -torch.abs(lat_vel)


def velocity_goal_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
) -> torch.Tensor:
    """Reward for velocity alignment towards goal."""
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]

    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    to_goal = torch.nn.functional.normalize(goal_pos - robot_pos, dim=1, eps=1e-6)

    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    return torch.sum(vel_w * to_goal, dim=1)


def goal_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward for making progress towards goal."""
    command = env.command_manager.get_command(command_name)
    dist = torch.norm(command[:, :2], dim=1)

    if "prev_dist" not in env.extras:
        env.extras["prev_dist"] = dist.clone()
        return torch.zeros_like(dist)

    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env.extras["prev_dist"][reset_mask] = dist[reset_mask]

    progress = env.extras["prev_dist"] - dist
    env.extras["prev_dist"] = dist.clone()
    return progress


def velocity_towards_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    stop_radius: float = 0.4,
) -> torch.Tensor:
    """Reward for velocity towards goal."""
    command = env.command_manager.get_command(command_name)
    rel_pos = command[:, :2]
    dist = torch.norm(rel_pos, dim=1)

    vel_b = env.scene["robot"].data.root_lin_vel_b[:, :2]
    to_goal = torch.nn.functional.normalize(rel_pos, dim=1, eps=1e-6)
    vel_proj = torch.sum(vel_b * to_goal, dim=1)

    return torch.where(dist > stop_radius, vel_proj, torch.zeros_like(vel_proj))


def heading_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward for heading alignment."""
    command = env.command_manager.get_command(command_name)
    cos_heading = command[:, 2]
    sin_heading = command[:, 3]

    heading_error = torch.atan2(sin_heading, cos_heading)
    return -torch.abs(heading_error)


def yaw_rate_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for yaw rate (rotation around z-axis)."""
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return -torch.abs(yaw_rate)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for joint velocity (L2 norm)."""
    joint_vel = env.scene["robot"].data.joint_vel
    return -torch.sum(joint_vel**2, dim=1)


def upright_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for upright orientation."""
    quat = env.scene["robot"].data.root_quat_w
    z_axis = torch.stack(
        [
            2 * (quat[:, 1] * quat[:, 3] - quat[:, 0] * quat[:, 2]),
            2 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1]),
            1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
        ],
        dim=1,
    )

    upright = z_axis[:, 2]
    return torch.clamp(upright, min=0.0)


def tilt_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for tilting from upright."""
    quat = env.scene["robot"].data.root_quat_w
    pitch = torch.asin(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]))
    roll = torch.atan2(
        2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]),
        1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2),
    )
    return -(pitch.abs() + roll.abs())


def velocity_heading_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Reward robot khi heading (hướng mặt) align với velocity command direction.

    Khuyến khích robot xoay về đúng hướng cần di chuyển trước khi đi.
    Đặc biệt quan trọng cho differential drive robot.

    Args:
        env: Environment instance
        command_name: Tên của velocity command
        std: Standard deviation cho exponential reward (nhỏ hơn = strict hơn)

    Returns:
        Reward tensor (0 khi heading sai 180°, 1 khi heading đúng)
    """
    # Lấy velocity command (vx, vy, wz)
    command = env.command_manager.get_command(command_name)
    cmd_vx = command[:, 0]  # Linear velocity x
    cmd_vy = command[:, 1]  # Linear velocity y (thường = 0 cho differential drive)
    cmd_wz = command[:, 2]  # Angular velocity z

    # Tính target heading từ velocity command
    # Target heading = hướng của velocity vector
    target_heading = torch.atan2(cmd_vy, cmd_vx)  # [-pi, pi]

    # Lấy current heading của robot từ quaternion
    quat = env.scene["robot"].data.root_quat_w
    # Extract yaw từ quaternion
    current_yaw = torch.atan2(
        2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )

    # Tính heading error (wrapped to [-pi, pi])
    heading_error = target_heading - current_yaw
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

    # Xác định khi nào cần check alignment:
    # - Khi có linear velocity command HOẶC
    # - Khi đang xoay để align (wz ≠ 0 và error lớn)
    velocity_magnitude = torch.sqrt(cmd_vx**2 + cmd_vy**2)
    has_linear_cmd = velocity_magnitude > 0.1  # Có command đi thẳng
    is_rotating = torch.abs(cmd_wz) > 0.1  # Đang có command xoay

    # Apply reward khi:
    # 1. Có linear command → check alignment
    # 2. Đang xoay VÀ heading error lớn → đang trong quá trình align
    should_check = has_linear_cmd | (is_rotating & (torch.abs(heading_error) > 0.3))

    # Exponential reward: exp(-|error|/std)
    # error = 0 → reward = 1.0
    # error = pi → reward = exp(-pi/std) ≈ 0 (nếu std=0.5)
    reward = torch.exp(-torch.abs(heading_error) / std)

    # Apply reward khi cần, otherwise return 1.0 (neutral)
    reward = torch.where(should_check, reward, torch.ones_like(reward))

    return reward
