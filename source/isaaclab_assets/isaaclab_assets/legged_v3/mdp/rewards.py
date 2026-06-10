"""Custom reward functions for Legged Robot V3."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Thưởng đứng thẳng — Gaussian kernel trên projected gravity.

    = 1.0 khi hoàn toàn thẳng đứng, decay về 0 khi nghiêng.
    Dùng với weight dương để tạo gradient bootstrap balance từ đầu training.
    std=0.3 rad: reward ≈ 0.37 khi nghiêng ~17°.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    grav_b = asset.data.projected_gravity_b   # (N, 3), unit vector
    tilt_sq = grav_b[:, 0] ** 2 + grav_b[:, 1] ** 2
    return torch.exp(-tilt_sq / (std ** 2))


def rpy_alignment_imu(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """L2 penalty cho roll+pitch từ IMU quaternion — YAW BỎ QUA.

    Trả về roll_err² + pitch_err² — dùng với weight ÂM.
    Yaw track riêng bởi track_ang_vel_z_world_exp.
    Port từ legged_v2: normalize quat, clamp, NaN-safe.
    """
    from isaaclab.utils.math import euler_xyz_from_quat
    imu = env.scene[imu_cfg.name]

    quat = imu.data.quat_w
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    if torch.isnan(quat).any() or torch.isinf(quat).any():
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)

    roll, pitch, _ = euler_xyz_from_quat(quat)
    roll  = torch.clamp(roll,  -torch.pi, torch.pi)
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)

    target_roll, target_pitch, _ = target_rpy
    roll_err  = torch.clamp(wrap_to_pi(roll  - target_roll),  -torch.pi, torch.pi)
    pitch_err = torch.clamp(wrap_to_pi(pitch - target_pitch), -torch.pi, torch.pi)

    penalty = torch.square(roll_err) + torch.square(pitch_err)
    return torch.nan_to_num(penalty, nan=0.0)


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


# ══════════════════════════════════════════════════════════════════════════════
# Velocity step-response quality
# ══════════════════════════════════════════════════════════════════════════════

def velocity_settling_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    band_vel: float = 0.10,
    band_yaw: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus khi vx_err VÀ yaw_rate_err đều nằm trong dải sai số nhỏ (đã ổn định).

    = 1.0 khi cả hai sai lệch trong band, = 0.0 khi ngoài.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vx_err  = cmd[:, 0] - asset.data.root_lin_vel_b[:, 0]
    yr_err  = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return (
        (torch.abs(vx_err) < band_vel) &
        (torch.abs(yr_err) < band_yaw)
    ).float()


def velocity_overshoot_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty khi vận tốc vượt qua setpoint (sign flip trên error).

    Trả về |error| tại thời điểm sign flip — dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vx_err = cmd[:, 0] - asset.data.root_lin_vel_b[:, 0]
    yr_err = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]

    prev_vx = getattr(env, "_prev_vx_err_sign", None)
    prev_yr = getattr(env, "_prev_yr_err_sign", None)
    sign_vx = torch.sign(vx_err)
    sign_yr = torch.sign(yr_err)
    env._prev_vx_err_sign = sign_vx.clone()
    env._prev_yr_err_sign = sign_yr.clone()

    if prev_vx is None:
        return torch.zeros(vx_err.shape[0], device=vx_err.device)

    vx_cross = (sign_vx * prev_vx) < 0
    yr_cross = (sign_yr * prev_yr) < 0
    return torch.abs(vx_err) * vx_cross.float() + torch.abs(yr_err) * yr_cross.float()


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
