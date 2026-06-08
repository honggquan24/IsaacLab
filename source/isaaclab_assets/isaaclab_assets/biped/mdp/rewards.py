"""Reward functions cho Biped inner tilt PID layer.

Đánh giá chất lượng đáp ứng bước nhảy (step response):
  tilt_tracking_exp  (+) : Gaussian kernel trên sai lệch roll/pitch
  yaw_tracking_exp   (+) : Gaussian kernel trên sai lệch yaw
  settling_bonus     (+) : bonus khi cả 3 sai lệch < ngưỡng
  overshoot_penalty  (-) : góc vượt qua target (sign flip)
  oscillation_penalty(-) : tốc độ góc đảo chiều khi gần target
  hip_torque_l2      (-) : năng lượng khớp hông
  wheel_torque_l2    (-) : năng lượng bánh xe
  action_rate_l2     (-) : gains thay đổi đột ngột
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _tilt_err(env, command_name, asset_cfg):
    from isaaclab.utils.math import euler_xyz_from_quat
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_err = torch.atan2(torch.sin(cmd[:, 2] - yaw), torch.cos(cmd[:, 2] - yaw))
    return cmd[:, 0] - roll, cmd[:, 1] - pitch, yaw_err, asset


# ══════════════════════════════════════════════════════════════════════════════
# Bám góc nghiêng — chất lượng đáp ứng
# ══════════════════════════════════════════════════════════════════════════════

def tilt_tracking_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian kernel trên roll+pitch error. = 1 khi bám hoàn hảo."""
    roll_err, pitch_err, _, _ = _tilt_err(env, command_name, asset_cfg)
    return torch.exp(-(roll_err**2 + pitch_err**2) / (std**2))


def yaw_tracking_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian kernel trên yaw error."""
    _, _, yaw_err, _ = _tilt_err(env, command_name, asset_cfg)
    return torch.exp(-(yaw_err**2) / (std**2))


def tilt_tracking_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 sai lệch roll+pitch (dùng với weight âm)."""
    roll_err, pitch_err, _, _ = _tilt_err(env, command_name, asset_cfg)
    return roll_err**2 + pitch_err**2


def settling_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    band_roll_pitch: float = 0.03,
    band_yaw:        float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus khi roll, pitch VÀ yaw đều nằm trong dải sai số nhỏ."""
    roll_err, pitch_err, yaw_err, _ = _tilt_err(env, command_name, asset_cfg)
    return (
        (torch.abs(roll_err)  < band_roll_pitch) &
        (torch.abs(pitch_err) < band_roll_pitch) &
        (torch.abs(yaw_err)   < band_yaw)
    ).float()


def overshoot_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty khi pitch hoặc roll vượt qua target (sign flip trên error)."""
    roll_err, pitch_err, _, _ = _tilt_err(env, command_name, asset_cfg)

    for attr, err in [("_prev_roll_sign", roll_err), ("_prev_pitch_sign", pitch_err)]:
        sign_now = torch.sign(err)
        if not hasattr(env, attr):
            setattr(env, attr, sign_now.clone())
        prev = getattr(env, attr)
        crossed = (sign_now * prev) < 0
        setattr(env, attr, sign_now.clone())

    roll_cross  = (torch.sign(roll_err)  * getattr(env, "_prev_roll_sign",  torch.sign(roll_err)))  < 0
    pitch_cross = (torch.sign(pitch_err) * getattr(env, "_prev_pitch_sign", torch.sign(pitch_err))) < 0
    return (torch.abs(roll_err) * roll_cross.float() +
            torch.abs(pitch_err) * pitch_cross.float())


def oscillation_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    near_band: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty khi tốc độ góc đảo chiều trong khi gần target (dao động)."""
    roll_err, pitch_err, _, asset = _tilt_err(env, command_name, asset_cfg)
    ang_vel = asset.data.root_ang_vel_b
    wx, wy  = ang_vel[:, 0], ang_vel[:, 1]

    for attr, w, err in [("_prev_wx_sign", wx, roll_err), ("_prev_wy_sign", wy, pitch_err)]:
        sign_now = torch.sign(w)
        if not hasattr(env, attr):
            setattr(env, attr, sign_now.clone())

    wx_sign = torch.sign(wx)
    wy_sign = torch.sign(wy)
    wx_rev  = (wx_sign * getattr(env, "_prev_wx_sign", wx_sign)) < 0
    wy_rev  = (wy_sign * getattr(env, "_prev_wy_sign", wy_sign)) < 0

    env._prev_wx_sign = wx_sign.clone()
    env._prev_wy_sign = wy_sign.clone()

    near_roll  = (torch.abs(roll_err)  < near_band).float()
    near_pitch = (torch.abs(pitch_err) < near_band).float()
    return near_roll * wx_rev.float() + near_pitch * wy_rev.float()


# ══════════════════════════════════════════════════════════════════════════════
# Smoothness / Energy
# ══════════════════════════════════════════════════════════════════════════════

def hip_torque_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 moment xoắn 4 khớp hông (dùng với weight âm)."""
    from isaaclab_assets.biped.biped_cfg import HIP_ALL_JOINT_NAMES
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset.find_joints(HIP_ALL_JOINT_NAMES)[0]
    return torch.sum(asset.data.applied_torque[:, ids] ** 2, dim=-1)


def wheel_torque_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 moment xoắn 2 bánh xe (dùng với weight âm)."""
    from isaaclab_assets.biped.biped_cfg import WHEEL_JOINT_NAMES
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset.find_joints(WHEEL_JOINT_NAMES)[0]
    return torch.sum(asset.data.applied_torque[:, ids] ** 2, dim=-1)


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """L2 tốc độ thay đổi gains giữa 2 bước (dùng với weight âm)."""
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1
    )


# ══════════════════════════════════════════════════════════════════════════════
# Outer loop: velocity tracking
# ══════════════════════════════════════════════════════════════════════════════

def velocity_tracking_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian trên sai lệch vx/vy. = 1 khi bám tốc độ hoàn hảo."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_lin_vel_b
    vx_err = cmd[:, 0] - lin_vel_b[:, 0]
    vy_err = cmd[:, 1] - lin_vel_b[:, 1]
    return torch.exp(-(vx_err**2 + vy_err**2) / (std**2))


def yaw_rate_tracking_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian trên sai lệch yaw_rate."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    yaw_rate = asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-((cmd[:, 2] - yaw_rate) ** 2) / (std**2))


def upright_exp(
    env: "ManagerBasedRLEnv",
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Thưởng đứng thẳng — projected gravity gần [0,0,-1]."""
    asset: Articulation = env.scene[asset_cfg.name]
    # projected_gravity = R^T * [0,0,-1] — khi thẳng đứng = [0,0,-9.81]
    grav_b = asset.data.projected_gravity_b   # (N, 3), normalized
    tilt_sq = grav_b[:, 0] ** 2 + grav_b[:, 1] ** 2
    return torch.exp(-tilt_sq / (std**2))


def rpy_alignment(
    env: "ManagerBasedRLEnv",
    std_roll: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Thưởng RPY alignment — phạt nghiêng thân theo góc roll (trục cân bằng Segway).

    Robot 2 bánh chỉ có bậc tự do roll → chỉ penalize roll.
    = 1.0 khi roll = 0 (đứng thẳng hoàn toàn).
    """
    from isaaclab.utils.math import euler_xyz_from_quat
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.exp(-(roll ** 2) / (std_roll ** 2))


def pitch_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phạt nghiêng trục pitch (Y) — Gaussian, = 1 khi pitch = 0."""
    from isaaclab.utils.math import euler_xyz_from_quat
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.exp(-(pitch ** 2) / (std ** 2))


def lin_vel_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 sai lệch vx/vy (dùng với weight âm)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_vel_b = asset.data.root_lin_vel_b
    return (cmd[:, 0] - lin_vel_b[:, 0]) ** 2 + (cmd[:, 1] - lin_vel_b[:, 1]) ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Outer loop: step-response quality (velocity)
# ══════════════════════════════════════════════════════════════════════════════

def velocity_settling_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    band_vel: float = 0.05,
    band_yaw: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus khi vy_err VÀ yaw_rate_err đều nằm trong dải sai số nhỏ."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vy_err  = cmd[:, 1] - asset.data.root_lin_vel_b[:, 1]
    yr_err  = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return (
        (torch.abs(vy_err) < band_vel) &
        (torch.abs(yr_err) < band_yaw)
    ).float()


def velocity_overshoot_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty khi vận tốc vượt qua setpoint (sign flip trên error)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vy_err = cmd[:, 1] - asset.data.root_lin_vel_b[:, 1]
    yr_err = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]

    for attr, err in [("_prev_vy_err_sign", vy_err), ("_prev_yr_err_sign", yr_err)]:
        sign_now = torch.sign(err)
        if not hasattr(env, attr):
            setattr(env, attr, sign_now.clone())
        prev = getattr(env, attr)
        setattr(env, attr, sign_now.clone())
        _ = prev  # mark used

    vy_cross = (torch.sign(vy_err) * getattr(env, "_prev_vy_err_sign", torch.sign(vy_err))) < 0
    yr_cross = (torch.sign(yr_err) * getattr(env, "_prev_yr_err_sign", torch.sign(yr_err))) < 0
    return torch.abs(vy_err) * vy_cross.float() + torch.abs(yr_err) * yr_cross.float()
