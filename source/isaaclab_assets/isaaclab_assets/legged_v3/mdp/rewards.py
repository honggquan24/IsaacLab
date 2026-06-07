"""Custom reward functions for Legged Robot V3."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rpy_alignment_imu(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    scale: float = 5.0
) -> torch.Tensor:
    """Penalty for deviating from the target roll/pitch (balance penalty).

    Returns -(roll_error² + pitch_error²), unbounded negative. Zero when perfectly
    upright, more negative as the robot tilts further from target.

    Args:
        env: The RL environment.
        target_rpy: Desired (roll, pitch, yaw) in radians.
        imu_cfg: Scene entity config for the IMU sensor.
    """
    imu = env.scene[imu_cfg.name]

    # Get and normalize quaternion
    quat = imu.data.quat_w
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    # Safety check
    if torch.isnan(quat).any() or torch.isinf(quat).any():
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    roll, pitch, yaw = euler_xyz_from_quat(quat)

    target_roll, target_pitch, _ = target_rpy  # ignore yaw for balance

    roll_error = wrap_to_pi(roll - target_roll)
    pitch_error = wrap_to_pi(pitch - target_pitch)

    total_error = torch.abs(roll_error) + torch.abs(pitch_error)
    return total_error


def equal_effort_leg_when_cmd(
    env: ManagerBasedRLEnv,
    command_name: str = "velocity_command",
    command_threshold: float = 0.05,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for asymmetric leg torques during active motion (wheels excluded).

    Compares each paired joint left-to-right: |left_hip - right_hip| + ...
    Returns the total absolute imbalance (positive) — use with a negative weight.
    Applied only when a velocity command is present (norm > command_threshold).

    Args:
        command_name: Name of the velocity command in CommandManager.
        command_threshold: Minimum command magnitude to activate the penalty.
        left_cfg: SceneEntityCfg whose joint_names list the LEFT leg joints (no wheel).
        right_cfg: SceneEntityCfg whose joint_names list the RIGHT leg joints (no wheel).
                   Joint order must correspond to left_cfg (hip → thigh → knee).
    """
    asset: Articulation = env.scene[left_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)

    cmd_norm = torch.norm(cmd[:, :2], dim=-1) + torch.abs(cmd[:, 2])
    has_cmd = (cmd_norm > command_threshold).float()  # (num_envs,)

    left_efforts  = torch.abs(asset.data.applied_torque[:, left_cfg.joint_ids])   # (num_envs, 3)
    right_efforts = torch.abs(asset.data.applied_torque[:, right_cfg.joint_ids])  # (num_envs, 3)
    penalty = torch.sum(torch.abs(left_efforts - right_efforts), dim=-1)           # (num_envs,)

    return has_cmd * penalty


def equal_effort_all_when_still(
    env: ManagerBasedRLEnv,
    command_name: str = "velocity_command",
    command_threshold: float = 0.05,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for asymmetric torques across all joints (including wheels) when standing still.

    Compares each paired joint left-to-right: hip + thigh + knee + wheel.
    Returns the total absolute imbalance (positive) — use with a negative weight.
    Applied only when the velocity command is below command_threshold (robot should be still).

    Args:
        command_name: Name of the velocity command in CommandManager.
        command_threshold: Maximum command magnitude that counts as "standing still".
        left_cfg: SceneEntityCfg whose joint_names list ALL LEFT joints (leg + wheel).
        right_cfg: SceneEntityCfg whose joint_names list ALL RIGHT joints (leg + wheel).
                   Joint order must correspond to left_cfg (hip → thigh → knee → wheel).
    """
    asset: Articulation = env.scene[left_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    cmd_norm = torch.norm(cmd[:, :2], dim=-1) + torch.abs(cmd[:, 2])
    is_still = (cmd_norm <= command_threshold).float()  # (num_envs,)

    left_efforts  = torch.abs(asset.data.applied_torque[:, left_cfg.joint_ids])   # (num_envs, 4)
    right_efforts = torch.abs(asset.data.applied_torque[:, right_cfg.joint_ids])  # (num_envs, 4)
    penalty = torch.sum(torch.abs(left_efforts - right_efforts), dim=-1)           # (num_envs,)

    return is_still * penalty


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for matching the commanded base height (Gaussian kernel).

    Reads target height from UniformPoseCommand[:, 2] (pos_z) and compares it
    to the robot root height in the world frame.

    Returns:
        Tensor shape (num_envs,) in range (0, 1].
        1.0 when height error is zero; decays toward 0 as error grows.

    Args:
        command_name: Key in CommandManager, e.g. "height_command".
        std: Sensitivity — reward ≈ 0.37 when |error| == std.
             std=0.05 m penalises errors larger than ~5 cm heavily.
        asset_cfg: Config of the robot articulation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # pos_z từ UniformPoseCommandCfg được lưu tại index 2 của command tensor
    target_height = env.command_manager.get_command(command_name)[:, 2]  # (num_envs,)
    current_height = asset.data.root_pos_w[:, 2]                         # (num_envs,)

    height_error_sq = torch.square(current_height - target_height)
    return torch.exp(-height_error_sq / (std ** 2))


def track_base_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared height-tracking error (use with a negative reward weight)."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_height  = env.command_manager.get_command(command_name)[:, 2]
    current_height = asset.data.root_pos_w[:, 2]
    return torch.square(current_height - target_height)
