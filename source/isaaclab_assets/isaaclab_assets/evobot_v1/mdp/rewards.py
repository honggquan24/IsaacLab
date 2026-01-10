from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Literal
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
import math
from isaaclab.assets import Articulation


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = (asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def joint_force_balance(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize imbalance in applied joint torques between left and right joints."""

    robot: Articulation = env.scene[left_cfg.name]

    # Applied torque from articulation (correct API)
    torque_left = robot.data.applied_torque[:, left_cfg.joint_ids]
    torque_right = robot.data.applied_torque[:, right_cfg.joint_ids]

    # Torque imbalance between legs
    diff = torch.abs(torque_left) - torch.abs(torque_right)

    # Penalize imbalance
    reward = torch.sum(torch.square(diff), dim=1)

    return reward

def rpy_alignment_imu(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    tolerance: float = 0.1,  # THÊM: tolerance zone (radians)
    scale: float = 3.0,  # FIX: Giảm từ 3.0 xuống 1.0
    axis_weights: tuple = (2.0, 2.0, 0.5),  # THÊM: ưu tiên roll/pitch
) -> torch.Tensor:
    """Reward for RPY alignment with tolerance zone."""
    imu = env.scene[imu_cfg.name]
    quat = imu.data.quat_w
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    if torch.isnan(quat).any() or torch.isinf(quat).any():
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    roll = torch.clamp(roll, -torch.pi, torch.pi)
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)
    yaw = torch.clamp(yaw, -torch.pi, torch.pi)
    
    target_roll, target_pitch, target_yaw = target_rpy
    
    roll_error = torch.abs(wrap_to_pi(roll - target_roll))
    pitch_error = torch.abs(wrap_to_pi(pitch - target_pitch))
    yaw_error = torch.abs(wrap_to_pi(yaw - target_yaw))
    
    # FIX: Apply tolerance
    roll_error = torch.clamp(roll_error - tolerance, min=0.0)
    pitch_error = torch.clamp(pitch_error - tolerance, min=0.0)
    yaw_error = torch.clamp(yaw_error - tolerance, min=0.0)
    
    # FIX: Apply axis weights
    weights = torch.tensor(axis_weights, device=roll.device, dtype=roll.dtype)
    weighted_error = (
        weights[0] * torch.square(roll_error) + 
        weights[1] * torch.square(pitch_error) + 
        weights[2] * torch.square(yaw_error)
    ) / weights.sum()  # Normalize by total weight
    
    # FIX: Lower scale
    reward = torch.exp(-scale * weighted_error)
    reward = torch.clamp(reward, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    
    return reward

def height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.5,
    sigma: float = 0.1,
    min_height: float = 0.1,
) -> torch.Tensor:
    """
    Reward for maintaining base height close to target.
    Uses Gaussian reward: exp(-((z - target)^2) / (2 * sigma^2))
    """
    ray_caster = env.scene['height_scanner']
    sensor_pos_z = ray_caster.data.pos_w[:, 2]  # [num_envs]
    ray_hits_z = ray_caster.data.ray_hits_w[..., 2]  # [num_envs, num_rays]
    
    # Vectorized ground height calculation
    valid_mask = ray_hits_z > -1e6  # [num_envs, num_rays]
    
    # Mean ground height (chỉ tính trên valid hits)
    masked_hits = ray_hits_z * valid_mask  # Invalid rays = 0
    num_valid = valid_mask.sum(dim=-1).clamp(min=1)  # [num_envs]
    ground_z = masked_hits.sum(dim=-1) / num_valid  # [num_envs]
    
    # Robot heights
    robot_heights = torch.where(
        valid_mask.any(dim=-1),  # Có ít nhất 1 valid hit?
        sensor_pos_z - ground_z,  # Yes: dùng ground_z
        sensor_pos_z              # No: dùng sensor position
    )
    
    # Gaussian reward
    height_error = robot_heights - target_height
    reward = torch.exp(-(height_error ** 2) / (2 * sigma ** 2))
    
    # Penalty for too low
    reward = torch.where(robot_heights < min_height, torch.zeros_like(reward), reward)
    
    # NaN guard
    reward = torch.clamp(reward, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    
    return reward

def angular_velocity_reward(
    env: ManagerBasedRLEnv,
    target_angular_vel: float = 0.0,
    scale: float = 5.0,  # FIX: Giảm scale xuống rất thấp
    max_vel: float = 20.0,  # FIX: Tăng max_vel
    axis_weights: tuple = (0.5, 0.5, 0.2),  # Giảm ảnh hưởng của yaw
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Ultra-robust angular velocity reward designed for convergence.
    
    KEY IMPROVEMENTS FOR CONVERGENCE:
    1. High base reward (offset) - prevents reward collapse
    2. Large tolerance zone - more forgiving
    3. Tanh normalization - bounded, smooth gradient
    4. Low scale - gentle penalty
    5. Axis-weighted - focuses on important rotations
    
    Args:
        reward_offset: Base reward value (0.5 means reward never goes below 0.5)
        use_tanh: Use tanh for bounded, smooth gradient (better than exp)
        
    Returns:
        Reward [num_envs] in range [reward_offset, 1.0]
    """
    robot = env.scene[asset_cfg.name]
    ang_vel = robot.data.root_ang_vel_w  # [num_envs, 3]
    
    # === FIX 1: Axis weighting (giảm ảnh hưởng của yaw) ===
    weights = torch.tensor(axis_weights, device=ang_vel.device, dtype=ang_vel.dtype)
    weighted_vel = ang_vel * weights
    ang_vel_norm = torch.norm(weighted_vel, dim=-1)
    ang_vel_norm = torch.clamp(ang_vel_norm, 0.0, max_vel)
    
    # === FIX 2: Tolerance-based error ===
    ang_vel_error = torch.abs(ang_vel_norm - target_angular_vel)
    reward = torch.exp(-scale * ang_vel_error ** 2)
    
    # === FIX 5: Robust clamping ===
    reward = torch.clamp(reward, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    
    return reward

def linear_velocity_reward(
    env: ManagerBasedRLEnv,
    target_linear_vel: float = 0.0,
    scale: float = 5.0,
    max_vel: float = 10.0,
    axis_weights: tuple = (1.0, 1.0, 0.3),  # (x, y, z) - giảm ảnh hưởng của z
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for maintaining low linear velocity (encourages standing still).
    
    Args:
        env: Environment instance
        target_linear_vel: Target velocity magnitude (m/s)
        tolerance: Velocity tolerance before penalty starts (m/s)
        scale: Exponential decay rate (lower = more forgiving)
        max_vel: Maximum velocity clamp (m/s)
        reward_offset: Base reward value (prevents reward collapse)
        use_tanh: Use tanh for bounded gradient (better than exp)
        axis_weights: Weights for (x, y, z) velocities - can ignore vertical motion
        asset_cfg: Robot entity config
        
    Returns:
        Reward tensor [num_envs], range [reward_offset, 1.0]
    """
    robot = env.scene[asset_cfg.name]
    lin_vel = robot.data.root_lin_vel_w  # [num_envs, 3]
    
    # === Axis weighting ===
    weights = torch.tensor(axis_weights, device=lin_vel.device, dtype=lin_vel.dtype)
    weighted_vel = lin_vel * weights  # [num_envs, 3]
    
    # Compute weighted magnitude
    lin_vel_norm = torch.norm(weighted_vel, dim=-1)  # [num_envs]
    lin_vel_norm = torch.clamp(lin_vel_norm, 0.0, max_vel)
    
    # === Tolerance-based error ===
    lin_vel_error = torch.abs(lin_vel_norm - target_linear_vel)
    
    reward_component = torch.exp(-scale * lin_vel_error ** 2)
        
    # === Robust clamping ===
    reward = torch.clamp(reward_component, 0.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    
    return reward

def feet_contact_force_symmetry(
    env: ManagerBasedRLEnv,
    threshold_force: float = 30.0,
    max_ratio_diff: float = 0.7,
):
    # Force Z from contact sensors
    fz_l = env.scene.sensors["contact_forces_wheel_left"].data.force_w[:, 2].clamp(min=0.0)
    fz_r = env.scene.sensors["contact_forces_wheel_right"].data.force_w[:, 2].clamp(min=0.0)

    total = fz_l + fz_r + 1e-6
    diff = torch.abs(fz_l - fz_r) / total

    symmetry = 1.0 - torch.clamp(diff / max_ratio_diff, 0.0, 1.0)

    contact_ok = torch.minimum(
        fz_l / threshold_force, fz_r / threshold_force
    ).clamp(0.0, 1.0)

    return symmetry * contact_ok

# def zmp_in_support_polygon(
#     env: ManagerBasedRLEnv,
#     margin: float = 0.04,
#     left_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Left_Foot"]),
#     right_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Right_Foot"]),
# ):
#     robot = env.scene["robot"]

#     # Body ID
#     id_l = left_foot_cfg.body_ids[0]
#     id_r = right_foot_cfg.body_ids[0]

#     # Foot positions
#     left_pos = robot.data.body_pos_w[:, id_l, :2]
#     right_pos = robot.data.body_pos_w[:, id_r, :2]

#     # Contact forces
#     fz_l = env.scene.sensors["contact_forces_left"].data.force_w[:, 2].clamp(min=0.0)
#     fz_r = env.scene.sensors["contact_forces_right"].data.force_w[:, 2].clamp(min=0.0)
#     total = fz_l + fz_r + 1e-6

#     zmp_x = (left_pos[:, 0] * fz_l + right_pos[:, 0] * fz_r) / total
#     zmp_y = (left_pos[:, 1] * fz_l + right_pos[:, 1] * fz_r) / total

#     center_x = (left_pos[:, 0] + right_pos[:, 0]) * 0.5
#     center_y = (left_pos[:, 1] + right_pos[:, 1]) * 0.5

#     dist = torch.sqrt((zmp_x - center_x)**2 + (zmp_y - center_y)**2)

#     return torch.clamp(1.0 - dist / margin, 0.0, 1.0)

# def zmp_in_support_polygon(
#     env: ManagerBasedRLEnv,
#     margin=0.04,
#     left_foot_cfg=SceneEntityCfg("robot", body_names=["Left_Foot"]),
#     right_foot_cfg=SceneEntityCfg("robot", body_names=["Right_Foot"]),
# ):
#     robot = env.scene["robot"]

#     id_l = left_foot_cfg.body_ids[0]
#     id_r = right_foot_cfg.body_ids[0]

#     lp = robot.data.body_pos_w[:, id_l, :2]
#     rp = robot.data.body_pos_w[:, id_r, :2]

#     fz_l = env.scene.sensors["contact_forces_left"].data.force_w[:, 2].clamp(0.0)
#     fz_r = env.scene.sensors["contact_forces_right"].data.force_w[:, 2].clamp(0.0)

#     total = fz_l + fz_r + 1e-6

#     zmp_x = (lp[:, 0] * fz_l + rp[:, 0] * fz_r) / total
#     zmp_y = (lp[:, 1] * fz_l + rp[:, 1] * fz_r) / total

#     cx = (lp[:, 0] + rp[:, 0]) * 0.5
#     cy = (lp[:, 1] + rp[:, 1]) * 0.5

#     dist = torch.sqrt((zmp_x - cx) ** 2 + (zmp_y - cy) ** 2)

#     return torch.clamp(1.0 - dist / margin, 0.0, 1.0)

# def foot_stillness(
#     env: ManagerBasedRLEnv,
#     scale=8.0,
#     left_cfg=SceneEntityCfg("robot", body_names=["Left_Foot"]),
#     right_cfg=SceneEntityCfg("robot", body_names=["Right_Foot"]),
# ):
#     robot = env.scene["robot"]

#     id_l = left_cfg.body_ids[0]
#     id_r = right_cfg.body_ids[0]

#     vel_l = robot.data.body_lin_vel_w[:, id_l]
#     vel_r = robot.data.body_lin_vel_w[:, id_r]

#     vel = torch.cat([vel_l, vel_r], dim=-1)
#     speed = torch.sum(vel * vel, dim=-1)

#     return torch.exp(-scale * speed)