"""Custom observation functions for Legged Robot V3."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sai lệch vận tốc [vx_err, yaw_rate_err]. Shape (N, 2).

    Giúp policy biết đang lệch bao nhiêu so với setpoint — có thể deploy
    được vì ước lượng từ encoder bánh xe + kinematic.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vx_err  = cmd[:, 0] - asset.data.root_lin_vel_b[:, 0]
    yr_err  = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return torch.stack([vx_err, yr_err], dim=-1)


def base_lin_vel_b(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vận tốc tịnh tiến body frame [vx, vy, vz]. Shape (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def wheel_angular_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tốc độ góc 2 bánh [left, right]. Shape (N, 2).

    Sign convention: positive = cả 2 bánh tiến về phía trước.
    Nếu robot xoay tại chỗ khi cùng positive velocity → flip _RIGHT_SIGN = -1.
    """
    _RIGHT_SIGN = 1  # đổi thành -1 nếu bánh phải quay ngược chiều
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_ids = asset.find_joints(["left_wheel_joint", "right_wheel_joint"])[0]
    vel = asset.data.joint_vel[:, wheel_ids].clone()
    vel[:, 1] *= _RIGHT_SIGN
    return vel


def base_height_w(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Chiều cao base_link so với mặt đất (world frame z). Shape (N, 1).

    Dùng trong sim để policy biết robot đang ở độ cao nào.
    Khi deploy thật: thay bằng kinematic_height_estimate() tính từ encoder.
    Spawn height = 0.383m. Min safe height ≈ 0.10m.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def kinematic_height_estimate(
    env: "ManagerBasedRLEnv",
    l_thigh: float = 0.15,
    l_shin: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ước tính chiều cao từ góc khớp (deployable — chỉ cần encoder).

    h ≈ l_thigh*cos(q_hip) + l_shin*cos(q_knee).
    Đây là xấp xỉ cho 5-bar linkage — điều chỉnh l_thigh/l_shin theo URDF thực tế.
    Shape (N, 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    hip_ids  = asset.find_joints(["left_hip_joint_A1",  "right_hip_joint_A1"])[0]
    knee_ids = asset.find_joints(["left_knee_joint_B1", "right_knee_joint_B1"])[0]

    q_hip  = asset.data.joint_pos[:, hip_ids]   # (N, 2)
    q_knee = asset.data.joint_pos[:, knee_ids]  # (N, 2)

    h_left  = l_thigh * torch.cos(q_hip[:, 0]) + l_shin * torch.cos(q_knee[:, 0])
    h_right = l_thigh * torch.cos(q_hip[:, 1]) + l_shin * torch.cos(q_knee[:, 1])
    return ((h_left + h_right) / 2.0).unsqueeze(-1)
