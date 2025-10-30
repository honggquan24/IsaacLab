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
    
    Returns:
        Reward tensor with shape (num_envs,)
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
    
    Args:
        env: Environment object.
        imu_cfg: SceneEntityCfg of the IMU sensor.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    imu = env.scene[imu_cfg.name]
    ang_vel = imu.data.ang_vel_b  # body frame angular velocity
    
    # Compute magnitude of angular velocity
    ang_speed = torch.norm(ang_vel, dim=-1)
    
    # Reward high when angular speed ≈ 0
    reward = torch.exp(-ang_speed**2 / 0.05)
    
    return reward


def upright_posture_reward(
    env: ManagerBasedEnv,
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    tolerance: float = 0.02,
) -> torch.Tensor:
    """
    Reward for keeping robot upright (roll and pitch near 0).
    
    Args:
        env: Environment object.
        imu_cfg: SceneEntityCfg of the IMU sensor.
        tolerance: Tolerance in radians for orientation error.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    imu = env.scene[imu_cfg.name]
    quat = imu.data.quat_w
    
    # Convert to Euler angles
    roll, pitch, _ = euler_xyz_from_quat(quat)
    
    # Calculate orientation error (only roll and pitch)
    error = torch.square(roll) + torch.square(pitch)
    
    # Exponential reward
    reward = torch.exp(-error / tolerance)
    
    return reward


def forward_velocity_reward(
    env: ManagerBasedEnv,
    target_velocity: float = 1.0,
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """
    Reward for moving forward at target velocity.
    
    Args:
        env: Environment object.
        target_velocity: Desired forward velocity in m/s.
        imu_cfg: SceneEntityCfg of the IMU sensor.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    imu = env.scene[imu_cfg.name]
    
    # Linear velocity in body frame
    lin_vel = imu.data.lin_vel_b
    
    # Forward velocity (x-axis in body frame)
    forward_vel = lin_vel[:, 0]
    
    # Velocity error
    vel_error = torch.square(forward_vel - target_velocity)
    
    # Reward
    reward = torch.exp(-vel_error / 0.5)
    
    return reward


def low_linear_acceleration_reward(
    env: ManagerBasedEnv,
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Reward for smooth motion (low linear acceleration).
    
    Args:
        env: Environment object.
        imu_cfg: SceneEntityCfg of the IMU sensor.
        threshold: Acceleration threshold for penalization.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    imu = env.scene[imu_cfg.name]
    
    # Linear acceleration in body frame
    lin_acc = imu.data.lin_acc_b
    
    # Magnitude of acceleration
    acc_magnitude = torch.norm(lin_acc, dim=-1)
    
    # Reward for low acceleration
    reward = torch.exp(-acc_magnitude / threshold)
    
    return reward


def balance_stability_reward(
    env: ManagerBasedEnv,
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    ang_vel_weight: float = 1.0,
    lin_acc_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined reward for balance stability using angular velocity and linear acceleration.
    
    Args:
        env: Environment object.
        imu_cfg: SceneEntityCfg of the IMU sensor.
        ang_vel_weight: Weight for angular velocity component.
        lin_acc_weight: Weight for linear acceleration component.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    imu = env.scene[imu_cfg.name]
    
    # Angular velocity magnitude
    ang_vel = imu.data.ang_vel_b
    ang_speed = torch.norm(ang_vel, dim=-1)
    
    # Linear acceleration magnitude
    lin_acc = imu.data.lin_acc_b
    acc_magnitude = torch.norm(lin_acc, dim=-1)
    
    # Combined reward
    ang_vel_reward = torch.exp(-ang_speed**2 / 0.1)
    lin_acc_reward = torch.exp(-acc_magnitude / 5.0)
    
    reward = ang_vel_weight * ang_vel_reward + lin_acc_weight * lin_acc_reward
    reward = reward / (ang_vel_weight + lin_acc_weight)
    
    return reward


def heading_alignment_reward(
    env: ManagerBasedEnv,
    target_heading: float = 0.0,
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """
    Reward for maintaining specific heading (yaw) direction.
    
    Args:
        env: Environment object.
        target_heading: Desired heading in radians.
        imu_cfg: SceneEntityCfg of the IMU sensor.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    imu = env.scene[imu_cfg.name]
    quat = imu.data.quat_w
    
    # Convert to Euler angles and extract yaw
    _, _, yaw = euler_xyz_from_quat(quat)
    
    # Heading error
    heading_error = wrap_to_pi(yaw - target_heading)
    
    # Reward
    reward = torch.exp(-torch.square(heading_error) / 0.2)
    
    return reward


def smooth_angular_motion_reward(
    env: ManagerBasedEnv,
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """
    Reward for smooth angular motion (penalizes sudden rotations).
    
    Args:
        env: Environment object.
        imu_cfg: SceneEntityCfg of the IMU sensor.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    imu = env.scene[imu_cfg.name]
    
    # Angular velocity components
    ang_vel = imu.data.ang_vel_b
    
    # Individual angular velocities
    roll_rate = torch.abs(ang_vel[:, 0])
    pitch_rate = torch.abs(ang_vel[:, 1])
    yaw_rate = torch.abs(ang_vel[:, 2])
    
    # Penalize high rates individually
    reward = torch.exp(-(roll_rate**2 + pitch_rate**2 + yaw_rate**2) / 0.3)
    
    return reward

def height_scanner_based_reward(
    env: ManagerBasedEnv,
    target_clearance: float = 0.35,
    scanner_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    tolerance: float = 0.05,
) -> torch.Tensor:
    """
    Reward for keeping the desired clearance from ground based on RayCaster data.
    
    Args:
        env: Environment object.
        target_clearance: Desired average distance from ground (m).
        scanner_cfg: SceneEntityCfg for the height scanner.
        tolerance: Smoothness of the reward curve.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    scanner = env.scene[scanner_cfg.name]
    
    # Dữ liệu quét (mỗi tia đo khoảng cách đến mặt đất)
    hit_distances = scanner.data.ray_hits_w[..., 2]  # hoặc .hit_distance nếu dùng API khác
    mean_height = torch.mean(hit_distances, dim=-1)
    
    # Sai số và phần thưởng
    height_error = mean_height - target_clearance
    reward = torch.exp(-torch.square(height_error) / tolerance)
    
    return reward
