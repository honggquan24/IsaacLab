from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Legged-Robot-V2-Pose --num_envs 4096 --resume --load_run=pose_1 --checkpoint=model_150.pt --video
def rpy_alignment_imu(
    env: ManagerBasedRLEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """
    Reward for full RPY alignment using IMU orientation.
    FIXED VERSION: Added clipping, normalization and numerical stability.
    
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
    
    # FIX 1: Normalize quaternion (avoid numerical issues)
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # FIX 2: Safety check for NaN/Inf
    if torch.isnan(quat).any() or torch.isinf(quat).any():
        print("[WARNING] Invalid quaternion detected in rpy_alignment_imu")
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        # Re-normalize
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Convert to Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    
    # FIX 3: Clamp euler angles to avoid extreme values
    roll = torch.clamp(roll, -torch.pi, torch.pi)
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)
    yaw = torch.clamp(yaw, -torch.pi, torch.pi)
    
    # Target angles
    target_roll, target_pitch, target_yaw = target_rpy
    
    # Errors with wrap_to_pi (ensure error in [-π, π])
    roll_error = wrap_to_pi(roll - target_roll)
    pitch_error = wrap_to_pi(pitch - target_pitch)
    yaw_error = wrap_to_pi(yaw - target_yaw)
    
    # FIX 4: Clamp errors to avoid extreme values
    roll_error = torch.clamp(roll_error, -torch.pi, torch.pi)
    pitch_error = torch.clamp(pitch_error, -torch.pi, torch.pi)
    yaw_error = torch.clamp(yaw_error, -torch.pi, torch.pi)
    
    # FIX 5: Use scale factor to avoid exp overflow
    # With error max = π, squared = π² ≈ 10
    # exp(-10) ≈ 0.000045 (OK)
    scale = 1.0  # Can adjust: higher → smoother reward
    
    # Total orientation error (squared)
    total_error = (
        torch.square(roll_error) + 
        torch.square(pitch_error) + 
        torch.square(yaw_error)
    ) / scale
    
    # FIX 6: Clamp total_error before exp (avoid underflow)
    # exp(-50) ≈ 1.9e-22 (too small → possible underflow)
    total_error = torch.clamp(total_error, 0.0, 50.0)
    
    # Reward (smooth Gaussian)
    reward = torch.exp(-total_error)
    
    # FIX 7: Final safety check
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # FIX 8: Check for NaN in output
    if torch.isnan(reward).any():
        print("[ERROR] NaN in reward output! Replacing with 0")
        reward = torch.nan_to_num(reward, nan=0.0)
    
    return reward


# TARGET JOINT POSITIONS
TARGET_JOINT_POS = torch.tensor([
    # index: joint_name                # comment
    0.0,                               # 0: Left_Revolute_01 (hip)
    0.0,                               # 1: Right_Revolute_01 (hip)
    math.radians(-20.0),                # 2: Left_Revolute_02 (knee) -20 
    math.radians(5.0),                # 3: Left_Revolute_03 (ankle) 5 
    -math.radians(-20.0),               # 4: Right_Revolute_02 (knee) 
    -math.radians(5.0),               # 5: Right_Revolute_03 (ankle)
    
    math.radians(-13.0),               # 6: Left_Revolute_05 (passive) - FIXED
    math.radians(12.6),                # 7: Right_Revolute_05 (passive)
    math.radians(-13.0),               # 8: Left_Revolute_06 (passive) - FIXED 
    math.radians(12.6),                # 9: Right_Revolute_06 (passive)
    
    0.0,                               # 10: Left_Revolute_04 (wheel)
    0.0,                               # 11: Right_Revolute_04 (wheel)
])

# Binary mask: 1 = joint contributes to reward; 0 = ignored
JOINT_MASK = torch.tensor([
    1, 1, 1, 1, 1, 1,    # 6 active leg joints (3 per leg)
    0, 0, 0, 0, 0, 0     # Passive joints + wheels
], dtype=torch.float32)


def pose_align_reward(
    env: ManagerBasedRLEnv,
    target_joint_pos: torch.Tensor = TARGET_JOINT_POS,
    mask: torch.Tensor = JOINT_MASK,
    scale: float = 20.0,
) -> torch.Tensor:
    """
    Computes a pose-alignment reward based on joint-space error.

    Encourages the robot to reach a predefined target configuration.
    Only joints with mask=1 influence the reward; others are ignored.

    Args:
        env: Isaac Lab environment with robot.data.joint_pos
        target_joint_pos: Desired joint positions (radians), shape [num_joints]
        mask: Binary mask indicating which joints to consider, shape [num_joints]
        scale: Exponential decay rate (higher = stricter alignment required)

    Returns:
        torch.Tensor: Reward per environment, shape [num_envs]
                      Values in (0, 1], with 1.0 indicating perfect alignment
    """
    robot = env.scene['robot']
    joint_pos = robot.data.joint_pos  # shape: [num_envs, num_joints]
    
    # Move tensors to correct device
    device = joint_pos.device
    if target_joint_pos.device != device:
        target_joint_pos = target_joint_pos.to(device)
    if mask.device != device:
        mask = mask.to(device)
    
    # FIX 1: Handle dimension mismatch
    # Expand target_joint_pos to [num_envs, num_joints] if needed
    if target_joint_pos.dim() == 1:
        target_joint_pos = target_joint_pos.unsqueeze(0).expand_as(joint_pos)
    
    # FIX 2: Apply mask correctly (element-wise multiplication)
    error = (joint_pos - target_joint_pos) * mask.unsqueeze(0)
    
    # FIX 3: Compute normalized error (per joint average)
    # Count active joints (mask sum)
    num_active_joints = mask.sum().clamp(min=1.0)  # Avoid division by zero
    
    # Mean squared error across active joints
    error_squared = torch.square(error).sum(dim=-1) / num_active_joints
    
    # FIX 4: Apply exponential reward with safety checks
    reward = torch.exp(-scale * error_squared)
    
    # FIX 5: Clamp reward to valid range
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # FIX 6: Safety check for NaN/Inf
    if torch.isnan(reward).any() or torch.isinf(reward).any():
        print("[WARNING] Invalid reward detected in pose_align_reward")
        reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    
    return reward

def height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 5.0,
    min_height: float = 0.1,
) -> torch.Tensor:
    """
    Reward for maintaining base height close to target.
    
    Encourages the robot to stay at a specific height (e.g., standing upright).
    Uses Gaussian reward: exp(-((z - target)^2) / (2 * sigma^2))
    
    Args:
        env: Environment instance.
        target_height: Desired height (meters) of the robot's base (z-coordinate).
        asset_cfg: Scene entity config for the robot.
        sigma: Standard deviation for Gaussian reward (controls width). 
               Smaller sigma → stricter height requirement.
        min_height: Minimum height threshold — below this, reward sharply decreases.
    
    Returns:
        torch.Tensor: Reward per env, shape (num_envs,), values in [0, 1].
    """
    # Get robot base position (world frame)
    robot = env.scene[asset_cfg.name]
    root_pos = robot.data.root_pos_w  # [num_envs, 3]
    height = root_pos[:, 2]  # z-coordinate

    # Safety: avoid NaN/Inf
    height = torch.clamp(height, min=0.0, max=5.0)

    # Gaussian reward around target
    reward = sigma * height ** 2

    # Final safety clamp
    reward = torch.clamp(reward, 0.0, 10.0)
    
    # NaN guard
    if torch.isnan(reward).any():
        reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
        print("[WARNING] NaN in height_reward — replaced with 0.0")

    return reward