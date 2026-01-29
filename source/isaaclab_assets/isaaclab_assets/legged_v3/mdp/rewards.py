"""Custom reward functions for Legged Robot V3."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rpy_alignment_imu(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """Reward for keeping the robot upright (roll/pitch alignment).

    Returns exp(-error²) in range (0, 1]. Higher = more upright.

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

    total_error = torch.square(roll_error) + torch.square(pitch_error)
    total_error = torch.clamp(total_error, 0.0, 50.0)

    return torch.exp(-total_error)
