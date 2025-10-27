from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def rpy_alignment_imu(
    env: ManagerBasedEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """
    Reward for full RPY alignment using IMU orientation.
    
    Args:
        env: Environment object.
        target_rpy: Desired (roll, pitch, yaw) in radians.
        imu_cfg: SceneEntityCfg of the IMU sensor.
    """
    # Get IMU sensor
    imu = env.scene[imu_cfg.name]

    # Quaternion orientation (world frame)
    quat = imu.data.quat_w 

    # Convert to Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quat)

    # Target angles
    target_roll, target_pitch, target_yaw = target_rpy

    # Errors
    roll_error = wrap_to_pi(roll - target_roll)
    pitch_error = wrap_to_pi(pitch - target_pitch)
    yaw_error = wrap_to_pi(yaw - target_yaw)

    # Total orientation error
    total_error = torch.square(roll_error) + torch.square(pitch_error) + torch.square(yaw_error)

    # Reward (smooth Gaussian)
    reward = torch.exp(-total_error / 0.3)

    return reward


def imu_stillness_reward(
    env: ManagerBasedEnv,
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """
    Reward for staying still (low angular velocity).
    """
    imu = env.scene[imu_cfg.name]
    ang_vel = imu.data.ang_vel_b  # body frame angular velocity

    # Compute magnitude of angular velocity
    ang_speed = torch.norm(ang_vel, dim=-1)

    # Reward high when angular speed ≈ 0
    reward = torch.exp(-ang_speed**2 / 0.05)
    return reward
                
                