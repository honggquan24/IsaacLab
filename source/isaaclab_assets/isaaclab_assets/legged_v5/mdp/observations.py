"""Custom observation functions for Legged Robot V3."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_rpy_deg(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    print_every: int = 200,
) -> torch.Tensor:
    """Roll/pitch/yaw của root link, đơn vị độ. Shape (N, 3). Dùng để debug orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    rpy = torch.stack([roll, pitch, yaw], dim=-1).rad2deg()
    if env.common_step_counter % print_every == 0:
        r0, p0, y0 = rpy[0, 0].item(), rpy[0, 1].item(), rpy[0, 2].item()
        h = asset.data.root_pos_w[0, 2].item()
        print(f"[DBG orient] step={env.common_step_counter:>7d}  h={h:.3f}  roll={r0:+.1f}°  pitch={p0:+.1f}°  yaw={y0:+.1f}°")
    return rpy


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


def com_pos_b(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vị trí khối tâm (mass-weighted CoM) trong body frame, tương đối với gốc base_link.
    Shape (N, 3).

    Hữu ích để policy biết CoM đang lệch về đâu so với base (trái/phải, trước/sau, cao/thấp).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_w   = asset.data.body_com_pos_w                            # (N, B, 3)
    masses  = asset.data.default_mass.to(com_w.device)            # (N, B)
    total_m = masses.sum(dim=-1, keepdim=True)                     # (N, 1)
    com_w_agg = (com_w * masses.unsqueeze(-1)).sum(dim=1) / total_m  # (N, 3)
    rel_w = com_w_agg - asset.data.root_pos_w
    return quat_apply_inverse(asset.data.root_quat_w, rel_w)


def com_to_wheel_plane_dist(
    env: "ManagerBasedRLEnv",
    wheel_right_body: str = "wheel_link_right",
    wheel_left_body:  str = "wheel_link_left",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Khoảng cách có dấu từ CoM đến mặt phẳng cân bằng (wheel balance plane).
    Shape (N, 1).

    Mặt phẳng cân bằng = mặt phẳng chứa trục bánh xe (axle) và phương thẳng đứng (Z).
    Khoảng cách dương = CoM lệch về phía bánh phải, âm = về phía bánh trái.
    Giá trị lý tưởng = 0 (CoM nằm đúng trên mặt phẳng giữa 2 bánh).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_names = asset.data.body_names
    ir = body_names.index(wheel_right_body)
    il = body_names.index(wheel_left_body)

    p_r = asset.data.body_pos_w[:, ir, :]  # (N, 3)
    p_l = asset.data.body_pos_w[:, il, :]  # (N, 3)
    mid = (p_r + p_l) * 0.5               # (N, 3) — tâm trục bánh

    # Hướng trục bánh (trái → phải), chuẩn hoá
    axle = p_r - p_l
    axle_unit = axle / axle.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Pháp tuyến mặt phẳng = axle × Z_world (hướng về phía trước robot)
    up = torch.zeros_like(axle_unit)
    up[:, 2] = 1.0
    normal = torch.linalg.cross(axle_unit, up)
    normal_unit = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Khối tâm mass-weighted
    com_w   = asset.data.body_com_pos_w
    masses  = asset.data.default_mass.to(com_w.device)
    total_m = masses.sum(dim=-1, keepdim=True)
    com_w_agg = (com_w * masses.unsqueeze(-1)).sum(dim=1) / total_m  # (N, 3)

    dist = ((com_w_agg - mid) * normal_unit).sum(dim=-1, keepdim=True)  # (N, 1)
    return dist


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
