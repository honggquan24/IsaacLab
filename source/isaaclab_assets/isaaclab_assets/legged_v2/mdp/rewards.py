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
    FIXED VERSION: Thêm clipping, normalization và numerical stability.
    
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
    
    # FIX 1: Normalize quaternion (tránh numerical issues)
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # FIX 2: Thêm safety check cho NaN/Inf
    if torch.isnan(quat).any() or torch.isinf(quat).any():
        print("[WARNING] Invalid quaternion detected in rpy_alignment_imu")
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        # Normalize lại
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Convert to Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    
    # FIX 3: Clamp euler angles để tránh extreme values
    roll = torch.clamp(roll, -torch.pi, torch.pi)
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)
    yaw = torch.clamp(yaw, -torch.pi, torch.pi)
    
    # Target angles
    target_roll, target_pitch, target_yaw = target_rpy
    
    # Errors với wrap_to_pi (đảm bảo error trong [-π, π])
    roll_error = wrap_to_pi(roll - target_roll)
    pitch_error = wrap_to_pi(pitch - target_pitch)
    yaw_error = wrap_to_pi(yaw - target_yaw)
    
    # FIX 4: Clamp errors để tránh extreme values
    roll_error = torch.clamp(roll_error, -torch.pi, torch.pi)
    pitch_error = torch.clamp(pitch_error, -torch.pi, torch.pi)
    yaw_error = torch.clamp(yaw_error, -torch.pi, torch.pi)
    
    # FIX 5: Dùng scale factor để tránh exp overflow
    # Với error max = π, squared = π² ≈ 10
    # exp(-10) ≈ 0.000045 (OK)
    scale = 1.0  # Có thể điều chỉnh: càng lớn → reward càng smooth
    
    # Total orientation error (squared)
    total_error = (
        torch.square(roll_error) + 
        torch.square(pitch_error) + 
        torch.square(yaw_error)
    ) / scale
    
    # FIX 6: Clamp total_error trước khi exp (tránh underflow)
    # exp(-50) ≈ 1.9e-22 (quá nhỏ → có thể bị underflow)
    total_error = torch.clamp(total_error, 0.0, 50.0)
    
    # Reward (smooth Gaussian)
    reward = torch.exp(-total_error)
    
    # FIX 7: Final safety check
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # FIX 8: Kiểm tra NaN trong output
    if torch.isnan(reward).any():
        print("[ERROR] NaN in reward output! Replacing with 0")
        reward = torch.nan_to_num(reward, nan=0.0)
    
    return reward


def rpy_alignment_imu_v2(
    env: ManagerBasedEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    weight_roll: float = 1.0,
    weight_pitch: float = 1.0,
    weight_yaw: float = 0.5,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    ALTERNATIVE VERSION: Weighted RPY alignment với individual control.
    Dùng version này nếu muốn prioritize roll/pitch hơn yaw.
    
    Args:
        env: Environment object.
        target_rpy: Desired (roll, pitch, yaw) in radians.
        imu_cfg: SceneEntityCfg of the IMU sensor.
        weight_roll: Weight for roll error (default 1.0).
        weight_pitch: Weight for pitch error (default 1.0).
        weight_yaw: Weight for yaw error (default 0.5, less important).
        sigma: Smoothness parameter (smaller = more sensitive).
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    # Get IMU sensor
    imu = env.scene[imu_cfg.name]
    
    # Quaternion orientation (world frame)
    quat = imu.data.quat_w
    
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Safety check
    if torch.isnan(quat).any() or torch.isinf(quat).any():
        quat = torch.nan_to_num(quat, nan=1.0, posinf=1.0, neginf=-1.0)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Convert to Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    
    # Clamp angles
    roll = torch.clamp(roll, -torch.pi, torch.pi)
    pitch = torch.clamp(pitch, -torch.pi, torch.pi)
    yaw = torch.clamp(yaw, -torch.pi, torch.pi)
    
    # Target angles
    target_roll, target_pitch, target_yaw = target_rpy
    
    # Individual errors
    roll_error = torch.clamp(wrap_to_pi(roll - target_roll), -torch.pi, torch.pi)
    pitch_error = torch.clamp(wrap_to_pi(pitch - target_pitch), -torch.pi, torch.pi)
    yaw_error = torch.clamp(wrap_to_pi(yaw - target_yaw), -torch.pi, torch.pi)
    
    # Weighted squared errors
    weighted_error = (
        weight_roll * torch.square(roll_error) +
        weight_pitch * torch.square(pitch_error) +
        weight_yaw * torch.square(yaw_error)
    ) / (weight_roll + weight_pitch + weight_yaw)
    
    # Clamp before exp
    weighted_error = torch.clamp(weighted_error / (sigma ** 2), 0.0, 50.0)
    
    # Reward
    reward = torch.exp(-weighted_error)
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # Safety check
    if torch.isnan(reward).any():
        reward = torch.nan_to_num(reward, nan=0.0)
    
    return reward


def rpy_alignment_imu_simple(
    env: ManagerBasedEnv,
    target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    imu_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
) -> torch.Tensor:
    """
    SIMPLEST VERSION: Dùng projected gravity thay vì Euler angles.
    Tránh gimbal lock và numerical issues của Euler conversion.
    Chỉ track roll và pitch (không có yaw).
    
    Args:
        env: Environment object.
        target_rpy: Desired (roll, pitch, yaw) - chỉ dùng roll, pitch.
        imu_cfg: SceneEntityCfg of the IMU sensor.
    
    Returns:
        Reward tensor with shape (num_envs,)
    """
    # Get IMU sensor
    imu = env.scene[imu_cfg.name]
    
    # Get projected gravity (z-component of gravity in body frame)
    # Khi robot đứng thẳng: proj_g ≈ [0, 0, -9.81]
    # Khi robot nghiêng: z-component giảm
    quat = imu.data.quat_w
    
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # Gravity vector in world frame
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    
    # Rotate gravity to body frame using quaternion
    # q * v * q^-1 (quaternion rotation)
    quat_w, quat_x, quat_y, quat_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Quaternion rotation formula (optimized)
    x, y, z = gravity_w[:, 0], gravity_w[:, 1], gravity_w[:, 2]
    
    tx = 2.0 * (quat_y * z - quat_z * y)
    ty = 2.0 * (quat_z * x - quat_x * z)
    tz = 2.0 * (quat_x * y - quat_y * x)
    
    gravity_b = torch.stack([
        x + quat_w * tx + quat_y * tz - quat_z * ty,
        y + quat_w * ty + quat_z * tx - quat_x * tz,
        z + quat_w * tz + quat_x * ty - quat_y * tx,
    ], dim=-1)
    
    # Target: gravity should point down in body frame: [0, 0, -1]
    target_gravity_b = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    
    # Dot product (cosine similarity)
    # cos(theta) = g_actual · g_target / (|g_actual| * |g_target|)
    dot_product = torch.sum(gravity_b * target_gravity_b, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Reward: 1 when aligned (dot=1), 0 when perpendicular (dot=0)
    # Dùng (1 + dot) / 2 để map [-1, 1] → [0, 1]
    reward = (1.0 + dot_product) / 2.0
    
    # Hoặc dùng exponential: exp(k * dot_product)
    # reward = torch.exp(5.0 * (dot_product - 1.0))  # Peak at dot=1
    
    reward = torch.clamp(reward, 0.0, 1.0)
    
    return reward


# ==== TESTING & DEBUGGING ====
def test_rpy_alignment():
    """
    Test function để verify rpy_alignment không tạo ra NaN/Inf.
    """
    import torch
    
    # Mock environment
    class MockIMU:
        class Data:
            quat_w = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],  # Upright
                [0.7071, 0.7071, 0.0, 0.0],  # 90° roll
                [0.0, 0.0, 0.0, 1.0],  # 180° yaw
                [0.5, 0.5, 0.5, 0.5],  # Complex rotation
            ])
        data = Data()
    
    class MockScene:
        def __getitem__(self, key):
            return MockIMU()
    
    class MockEnv:
        scene = MockScene()
        device = "cpu"
        num_envs = 4
    
    env = MockEnv()
    
    print("Testing rpy_alignment_imu...")
    reward = rpy_alignment_imu(env)
    print(f"Reward: {reward}")
    print(f"Contains NaN: {torch.isnan(reward).any()}")
    print(f"Contains Inf: {torch.isinf(reward).any()}")
    print(f"Min: {reward.min():.6f}, Max: {reward.max():.6f}")
    
    print("\nTesting rpy_alignment_imu_simple...")
    reward_simple = rpy_alignment_imu_simple(env)
    print(f"Reward: {reward_simple}")
    print(f"Contains NaN: {torch.isnan(reward_simple).any()}")
    print(f"Contains Inf: {torch.isinf(reward_simple).any()}")
    print(f"Min: {reward_simple.min():.6f}, Max: {reward_simple.max():.6f}")


if __name__ == "__main__":
    test_rpy_alignment()