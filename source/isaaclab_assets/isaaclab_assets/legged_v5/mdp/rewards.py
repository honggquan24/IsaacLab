"""Custom reward functions for Legged Robot V3."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi, quat_mul, euler_xyz_from_quat, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rpy_alignment(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    roll_scale: float = 1.0,
    pitch_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty roll/pitch lệch so với orientation mặc định (init_state.rot).

    Tính q_rel = q_ref^{-1} * q_cur, rồi euler_xyz → roll, pitch.
    Trả về roll_scale*roll² + pitch_scale*pitch² — dùng với weight ÂM.

    roll_scale/pitch_scale cho phép phạt KHÔNG đối xứng: với robot bánh kiểu
    Segway, nghiêng dọc (lean) là cần thiết để di chuyển → để pitch_scale nhỏ;
    nghiêng ngang (đổ sang bên) phải tránh → roll_scale lớn.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_cur = asset.data.root_quat_w                          # (N, 4) w,x,y,z
    quat_ref = asset.data.default_root_state[:, 3:7]           # (N, 4)

    quat_ref_inv = quat_ref * torch.tensor([1, -1, -1, -1], device=quat_ref.device)
    quat_rel = quat_mul(quat_ref_inv, quat_cur)

    roll, pitch, _ = euler_xyz_from_quat(quat_rel)
    roll  = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)

    return roll_scale * torch.square(roll) + pitch_scale * torch.square(pitch)


def track_lin_vel_x_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track CHỈ vận tốc trục X (frame yaw, exp kernel) — tách từ track_lin_vel_xy để debug.

    Giống track_lin_vel_xy_yaw_frame_exp nhưng chỉ tính sai số thành phần X (index 0).
    Cho phép xem riêng robot bám trục X tốt tới đâu trên log.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    error = torch.square(env.command_manager.get_command(command_name)[:, 0] - vel_yaw[:, 0])
    return torch.exp(-error / std**2)


def track_lin_vel_y_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track CHỈ vận tốc trục Y (frame yaw, exp kernel) — tách từ track_lin_vel_xy để debug.

    Giống track_lin_vel_xy_yaw_frame_exp nhưng chỉ tính sai số thành phần Y (index 1).
    Với V5, lin_vel_y = hướng tiến → reward này phản ánh trực tiếp khả năng đi tới/lui.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    error = torch.square(env.command_manager.get_command(command_name)[:, 1] - vel_yaw[:, 1])
    return torch.exp(-error / std**2)


def lateral_tilt_penalty(
    env: ManagerBasedRLEnv,
    axis: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phạt nghiêng NGANG (lateral) — chỉ thành phần gravity theo trục lateral của body.

    Robot drive theo world-Y (đã xoay 90° quanh X). Probe projected_gravity_b:
      - axis 0 (body-X) thay đổi khi nghiêng ngang  → PHẠT (đổ sang bên).
      - axis 2 (body-Z) thay đổi khi lean dọc fore-aft → KHÔNG phạt (cần để chạy theo Y).
    Trả về g_axis² ∈ [0,1] — dùng với weight ÂM. Không phụ thuộc thứ tự euler/offset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, axis])


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
    # Chiều tiến của robot là body-Y (thân xoay 90° quanh X) → dùng index 1.
    vy_err  = cmd[:, 1] - asset.data.root_lin_vel_b[:, 1]
    yr_err  = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return (
        (torch.abs(vy_err) < band_vel) &
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
    # Chiều tiến của robot là body-Y (thân xoay 90° quanh X) → dùng index 1.
    vy_err = cmd[:, 1] - asset.data.root_lin_vel_b[:, 1]
    yr_err = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]

    prev_vy  = getattr(env, "_prev_vy_err_sign", None)
    prev_yr  = getattr(env, "_prev_yr_err_sign", None)
    prev_cmd = getattr(env, "_prev_overshoot_cmd", None)
    sign_vy = torch.sign(vy_err)
    sign_yr = torch.sign(yr_err)
    env._prev_vy_err_sign  = sign_vy.clone()
    env._prev_yr_err_sign  = sign_yr.clone()
    env._prev_overshoot_cmd = cmd.clone()

    if prev_vy is None:
        return torch.zeros(vy_err.shape[0], device=vy_err.device)

    # Bỏ qua sign flip do setpoint đổi (command resample) hoặc episode vừa reset —
    # đó không phải overshoot thật.
    cmd_unchanged = (cmd - prev_cmd).abs().sum(dim=-1) < 1e-6
    not_fresh     = env.episode_length_buf > 1
    gate = (cmd_unchanged & not_fresh).float()

    vy_cross = (sign_vy * prev_vy) < 0
    yr_cross = (sign_yr * prev_yr) < 0
    penalty = torch.abs(vy_err) * vy_cross.float() + torch.abs(yr_err) * yr_cross.float()
    return penalty * gate


def com_wheel_plane_alignment(
    env: ManagerBasedRLEnv,
    wheel_right_body: str = "wheel_link_right",
    wheel_left_body: str = "wheel_link_left",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phạt khoảng cách từ CoM toàn robot đến mặt phẳng đứng chứa tâm 2 bánh xe.

    CoM = mass-weighted trung bình của body_com_pos_w (CoM thật, dịch theo tư thế chân).
    Mặt phẳng được xác định bởi: đường nối tâm 2 bánh xe + trục đứng (Z).
    Normal = normalize((P_R - P_L) × Z_world) — vector ngang, vuông góc trục bánh.
    Trả về |dist| — dùng với weight âm.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve body index + mass 1 lần, cache trên env
    cache = getattr(env, "_com_plane_cache", None)
    if cache is None:
        right_idx = asset.data.body_names.index(wheel_right_body)
        left_idx  = asset.data.body_names.index(wheel_left_body)
        masses = asset.data.default_mass.to(asset.device).unsqueeze(-1)  # (N, B, 1)
        cache = (right_idx, left_idx, masses, masses.sum(dim=1))
        env._com_plane_cache = cache
    right_idx, left_idx, masses, total_mass = cache

    # CoM toàn robot: mass-weighted trên CoM từng body
    com = (asset.data.body_com_pos_w * masses).sum(dim=1) / total_mass   # (N, 3)

    # Tâm 2 bánh xe (link origin = trục quay bánh)
    p_r = asset.data.body_pos_w[:, right_idx, :]    # (N, 3)
    p_l = asset.data.body_pos_w[:, left_idx,  :]    # (N, 3)

    # Vector nối 2 bánh xe
    axle = p_r - p_l                                # (N, 3)

    # Z world
    z = torch.zeros_like(axle)
    z[:, 2] = 1.0

    # Normal của mặt phẳng = axle × Z (vuông góc với cả axle và Z)
    n = torch.cross(axle, z, dim=-1)                # (N, 3)
    n_norm = n.norm(dim=-1, keepdim=True)
    normal = n / n_norm.clamp(min=1e-6)

    # Midpoint của 2 bánh xe (điểm thuộc mặt phẳng)
    mid = (p_r + p_l) * 0.5                         # (N, 3)

    # Khoảng cách có dấu từ CoM đến mặt phẳng
    dist = torch.abs(((com - mid) * normal).sum(dim=-1))   # (N,)

    # Degenerate: trục bánh gần thẳng đứng (robot ngã 90°) → normal vô nghĩa → trả 0
    valid = (n_norm.squeeze(-1) > 1e-3).float()
    return dist * valid


def hip_symmetry_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["right_hip_joint", "left_hip_joint"]
    ),
) -> torch.Tensor:
    """Phạt 2 chân lệch nhau (mất đối xứng trái-phải).

    Stance đối xứng: right_hip và left_hip là ảnh gương → right_hip ≈ -left_hip,
    nên (right_hip + left_hip) ≈ 0. Trả về bình phương tổng — dùng với weight ÂM.
    Bất biến với thứ tự joint_ids vì dùng (a+b)².
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jpos = asset.data.joint_pos[:, asset_cfg.joint_ids]   # (N, 2)
    return torch.square(jpos[:, 0] + jpos[:, 1])


def track_base_height_above_axle_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.04,
    wheel_right_body: str = "wheel_link_right",
    wheel_left_body: str = "wheel_link_left",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward track độ cao base TƯƠNG ĐỐI so với trục bánh xe (Gaussian kernel).

    height = z_base − z_axle_mid, với z_axle_mid = trung bình z tâm 2 bánh.
    Không phụ thuộc terrain — hoạt động đúng cả trên địa hình gồ ghề,
    khác track_base_height_exp (đo z tuyệt đối world, chỉ đúng trên mặt phẳng).

    Target đọc từ UniformPoseCommand[:, 2] (pos_z) — hiểu là độ cao tương đối.
    Trả về (0, 1]: 1.0 khi đúng độ cao, ≈0.37 khi |error| == std.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    cache = getattr(env, "_axle_height_cache", None)
    if cache is None:
        cache = (
            asset.data.body_names.index(wheel_right_body),
            asset.data.body_names.index(wheel_left_body),
        )
        setattr(env, "_axle_height_cache", cache)
    right_idx, left_idx = cache

    z_axle = (
        asset.data.body_pos_w[:, right_idx, 2] + asset.data.body_pos_w[:, left_idx, 2]
    ) * 0.5
    height = asset.data.root_pos_w[:, 2] - z_axle               # (N,)

    target = env.command_manager.get_command(command_name)[:, 2]  # (N,)
    return torch.exp(-torch.square(height - target) / (std ** 2))


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


def wheel_air_time_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Phạt tỉ lệ thời gian bánh xe bay khỏi mặt đất.

    Giống feet_air_time nhưng KHÔNG gate theo velocity command — bánh xe
    phải tiếp đất mọi lúc, kể cả khi robot đứng yên.
    Trả về accumulated air time tại thời điểm bánh chạm đất lần đầu,
    dùng với weight âm.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.sum((last_air_time - threshold) * first_contact, dim=1)
