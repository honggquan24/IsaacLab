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
    """Phạt khớp chân 2 bên xuất lực không đều nhau khi đang chạy (loại trừ bánh xe).

    "Đều nhau" = tổng lực chân trái ≈ tổng lực chân phải.
    So sánh từng khớp tương ứng: |left_hip - right_hip| + |left_thigh - right_thigh| + ...
    Trả về tổng sai lệch (dương) → dùng với weight âm.

    Args:
        command_name: Tên velocity command.
        command_threshold: Ngưỡng để xác định "có lệnh".
        left_cfg: SceneEntityCfg với joint_names = các khớp chân TRÁI (không gồm bánh xe).
        right_cfg: SceneEntityCfg với joint_names = các khớp chân PHẢI (không gồm bánh xe).
                   Thứ tự khớp phải tương ứng với left_cfg (hip→thigh→knee).
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
    """Phạt tất cả khớp 2 bên (kể cả bánh xe) xuất lực không đều nhau khi đứng yên.

    "Đều nhau" = tổng lực chân trái ≈ tổng lực chân phải (kể cả bánh xe).
    So sánh từng khớp tương ứng: hip + thigh + knee + wheel.
    Trả về tổng sai lệch (dương) → dùng với weight âm.

    Args:
        command_name: Tên velocity command.
        command_threshold: Ngưỡng để xác định "dừng".
        left_cfg: SceneEntityCfg với joint_names = tất cả khớp bên TRÁI (leg + wheel).
        right_cfg: SceneEntityCfg với joint_names = tất cả khớp bên PHẢI (leg + wheel).
                   Thứ tự khớp phải tương ứng với left_cfg (hip→thigh→knee→wheel).
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
    """Reward robot đứng đúng độ cao được yêu cầu (exp-kernel).

    Đọc target height từ UniformPoseCommand[:, 2] (pos_z) và so sánh
    với độ cao thực tế của root trong world frame.

    Returns:
        Tensor (num_envs,) trong khoảng (0, 1].
        = 1.0 khi height_error = 0, giảm dần khi sai lệch tăng.

    Args:
        command_name: Tên command trong CommandManager (ví dụ "height_command").
        std:          Độ nhạy — sai lệch bằng std thì reward ≈ 0.37.
                      std=0.05 m → phạt nặng khi sai > 5 cm.
        asset_cfg:    Config của robot asset.
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
    """Penalty L2 cho sai lệch chiều cao (trả về giá trị âm, dùng weight âm)."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_height  = env.command_manager.get_command(command_name)[:, 2]
    current_height = asset.data.root_pos_w[:, 2]
    return torch.square(current_height - target_height)
