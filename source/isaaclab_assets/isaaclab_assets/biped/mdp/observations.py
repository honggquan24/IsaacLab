"""Observation functions cho Biped inner tilt PID layer.

Obs 74-dim:
    tilt_error(3)       = [roll_err, pitch_err, yaw_err]
    imu_quat(4)         = quaternion orientation [w,x,y,z]
    imu_lin_acc_b(3)    = linear acceleration body frame
    ang_vel_b(3)        = [wx, wy, wz]  (IMU gyro)
    projected_gravity(3)= [gx, gy, gz]
    hip_pos_error(4)    = q_des_hip - q_hip
    hip_vel(4)          = dq_hip
    wheel_vel(2)        = [omega_L, omega_R]
    all_joint_pos(10)   = tất cả joint positions (relative to default)
    all_joint_vel(10)   = tất cả joint velocities
    all_joint_acc(10)   = tất cả joint accelerations
    last_action(21)     = 7 × 3 gains (từ action manager)
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tilt_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """[roll_err, pitch_err, yaw_err] — sai lệch giữa cmd_tilt và góc thực. Shape (N, 3)."""
    from isaaclab.utils.math import euler_xyz_from_quat
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)   # (N, 3)
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_err = torch.atan2(torch.sin(cmd[:, 2] - yaw), torch.cos(cmd[:, 2] - yaw))
    return torch.stack([cmd[:, 0] - roll, cmd[:, 1] - pitch, yaw_err], dim=-1)


def hip_pos_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    roll_alloc:  tuple = ( 0.5, -0.5,  0.5, -0.5),
    pitch_alloc: tuple = ( 0.3,  0.3,  0.3,  0.3),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """q_des_hip - q_hip cho 4 khớp hông. Shape (N, 4)."""
    from isaaclab_assets.biped.biped_cfg import HIP_ALL_JOINT_NAMES, HIP_DEFAULT_Q
    asset: Articulation = env.scene[asset_cfg.name]
    hip_ids = asset.find_joints(HIP_ALL_JOINT_NAMES)[0]
    q_hip   = asset.data.joint_pos[:, hip_ids]

    cmd = env.command_manager.get_command(command_name)   # (N, 3)
    ra = torch.tensor(roll_alloc,  device=env.device, dtype=torch.float32)
    pa = torch.tensor(pitch_alloc, device=env.device, dtype=torch.float32)
    qd = torch.tensor(HIP_DEFAULT_Q, device=env.device, dtype=torch.float32)

    q_des = qd + cmd[:, 0:1] * ra + cmd[:, 1:2] * pa   # (N, 4)
    return q_des - q_hip


def hip_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vận tốc 4 khớp hông. Shape (N, 4)."""
    from isaaclab_assets.biped.biped_cfg import HIP_ALL_JOINT_NAMES
    asset: Articulation = env.scene[asset_cfg.name]
    hip_ids = asset.find_joints(HIP_ALL_JOINT_NAMES)[0]
    return asset.data.joint_vel[:, hip_ids]


def wheel_angular_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tốc độ góc 2 bánh [left, right]. Shape (N, 2)."""
    from isaaclab_assets.biped.biped_cfg import WHEEL_JOINT_NAMES
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_ids = asset.find_joints(WHEEL_JOINT_NAMES)[0]
    return asset.data.joint_vel[:, wheel_ids]


# ── IMU ───────────────────────────────────────────────────────────────────────

def imu_quat(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quaternion orientation [w, x, y, z] của thân robot. Shape (N, 4)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def imu_lin_acc_b(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gia tốc tịnh tiến trong body frame [ax, ay, az]. Shape (N, 3).

    Tính từ gia tốc tâm khối của root body (world frame) xoay về body frame.
    """
    from isaaclab.utils.math import quat_apply_inverse
    asset: Articulation = env.scene[asset_cfg.name]
    acc_w = asset.data.body_com_lin_acc_w[:, 0, :]   # root body, world frame (N, 3)
    return quat_apply_inverse(asset.data.root_quat_w, acc_w)


# ── Tất cả joints ─────────────────────────────────────────────────────────────

def all_joint_pos_rel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vị trí tất cả joints (relative to default). Shape (N, N_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def all_joint_vel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vận tốc tất cả joints. Shape (N, N_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def all_joint_acc(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gia tốc tất cả joints (finite diff). Shape (N, N_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc


# ── Outer loop observations ────────────────────────────────────────────────────

def base_lin_vel_b(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vận tốc tịnh tiến trong body frame [vx, vy, vz]. Shape (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def velocity_command(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Velocity setpoint từ command manager [vx_des, vy_des, yaw_rate_des]. Shape (N, 3)."""
    return env.command_manager.get_command(command_name)


def velocity_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sai lệch vận tốc [vx_err, vy_err, yaw_rate_err]. Shape (N, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_lin_vel_b
    ang_vel_b = asset.data.root_ang_vel_b
    return torch.stack([
        cmd[:, 0] - lin_vel_b[:, 0],
        cmd[:, 1] - lin_vel_b[:, 1],
        cmd[:, 2] - ang_vel_b[:, 2],
    ], dim=-1)
