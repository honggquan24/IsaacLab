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
    """Tốc độ góc 2 bánh [left, right]. Shape (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_ids = asset.find_joints(["left_wheel_joint", "right_wheel_joint"])[0]
    return asset.data.joint_vel[:, wheel_ids]
