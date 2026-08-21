# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manipulation and velocity-specific reward functions for Evobot V1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =========================================================
# VELOCITY-SPECIFIC REWARDS
# =========================================================


def reward_wheel_speed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for controlled wheel speed (velocity tracking)."""
    commands = env.command_manager.get_command("base_velocity")
    target_lin_vel = commands[:, :2]
    robot = env.scene["robot"]
    actual_lin_vel = robot.data.root_lin_vel_w[:, :2]
    vel_error = torch.norm(actual_lin_vel - target_lin_vel, dim=-1)
    reward = torch.exp(-5.0 * vel_error)
    return reward


# =========================================================
# LOCOMOTION-MANIPULATION UTILITIES
# =========================================================


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action changes for smooth control."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel

    if not hasattr(asset.data, "prev_joint_vel"):
        asset.data.prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    joint_acc = (joint_vel - asset.data.prev_joint_vel) / env.physics_dt
    asset.data.prev_joint_vel = joint_vel.clone()

    return torch.sum(torch.square(joint_acc), dim=1)


def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Phạt phần lực va chạm vượt ``threshold`` [N] trên các link được chọn.

    Chỉ tính trên ``sensor_cfg.body_ids`` rồi cộng theo link, nên trả về shape
    ``(num_envs,)`` đúng như reward manager yêu cầu.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.sum((net_contact_force - threshold).clamp(min=0.0), dim=1)


def reward_man(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tanh-based position tracking reward for manipulation targets.

    Rewards the agent for moving the end-effector towards the commanded position.
    """
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]

    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]

    pos_error = torch.norm(ee_pos - target_pos, dim=-1)
    reward = torch.tanh(pos_error / std).unsqueeze(-1).squeeze(-1)
    reward = 1.0 - reward

    return reward


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def gripper_height_tracking_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize gripper height error from command (z-position tracking).

    Tracks the relative z-position of the gripper in the robot's base frame.
    Command z-component specifies target height relative to base.

    Args:
        env: Environment instance
        command_name: Name of the pose command (we extract z from it)
        asset_cfg: Asset configuration with body_ids pointing to gripper link

    Returns:
        L2 error between current gripper height and command z-target
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Lệnh có dạng [x, y, z, qw, qx, qy, qz]; chỉ lấy z làm độ cao mong muốn [m].
    target_z = command[:, 2].unsqueeze(-1)

    # Độ cao kẹp so với gốc thân robot [m] — dùng body_ids đúng như docstring mô tả.
    # (Trước đây hàm đọc nhầm ``joint_pos`` nên trả về sai chiều và sai đại lượng.)
    gripper_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(-1)

    return torch.sum(torch.square(gripper_z - target_z), dim=1)


def binary_gripper_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    joint_limits: tuple[float, float] = (0.0, 0.04),
) -> torch.Tensor:
    """Reward for tracking binary gripper commands (0=close, 1=open).

    Normalizes joint position to [0, 1] range and compares with binary command.
    Uses exponential reward for smooth gradients.

    Args:
        env: Environment instance
        command_name: Name of the binary gripper command
        asset_cfg: Asset configuration with joint_ids pointing to gripper joint
        joint_limits: (min, max) physical limits of gripper joint (default: 0.0 to 0.04m)

    Returns:
        Exponential reward based on tracking error (higher is better)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Extract binary target from command [x, y, z, qw, qx, qy, qz]
    # Binary value is at index 2 (z position): 0.0 = close, 1.0 = open
    binary_target = command[:, 2]  # Shape: (num_envs,)

    # Get current joint position (prismatic joint, typically in meters)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)  # Shape: (num_envs,)

    # Normalize joint position to [0, 1] range
    joint_min, joint_max = joint_limits
    normalized_pos = (joint_pos - joint_min) / (joint_max - joint_min)
    normalized_pos = torch.clamp(normalized_pos, 0.0, 1.0)

    # Compute tracking error (how far from binary target)
    error = torch.abs(normalized_pos - binary_target)

    # Exponential reward: exp(-error/std) - higher when error is small
    # std=0.2 means 90% reward at 20% error, which is reasonable for binary control
    reward = torch.exp(-error / 0.2)

    return reward


def joint_angle_command_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position error from command (using yaw component as joint target).

    This function extracts the 'yaw' component from a UniformPoseCommand and uses it
    as the target joint angle for revolute joints.

    Args:
        env: Environment instance
        command_name: Name of the pose command (we extract yaw from it)
        asset_cfg: Asset configuration with joint_names or joint_ids

    Returns:
        L2 error between current joint position and command yaw
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Extract quaternion from pose command [x, y, z, qw, qx, qy, qz]
    target_quat = command[:, 3:7]  # Shape: (num_envs, 4)

    # Convert quaternion to euler angles (returns tuple: roll, pitch, yaw)
    _, _, yaw = euler_xyz_from_quat(target_quat)  # yaw: (num_envs,)

    # Get current joint positions
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])  # Shape: (num_envs, num_joints)

    # Wrap target yaw to (-pi, pi)
    target_yaw_wrapped = wrap_to_pi(yaw).unsqueeze(-1)  # Shape: (num_envs, 1)

    # Compute L2 error
    return torch.sum(torch.square(joint_pos - target_yaw_wrapped), dim=1)
