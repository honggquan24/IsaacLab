"""Navigation reward functions cho chính sách phân cấp Legged V5.

QUAN TRỌNG — hệ trục lệnh:
`UniformPose2dCommand.command` trả `[pos_command_b_x, pos_command_b_y, heading_b]`
ở **body yaw-frame** (đã là vector robot→đích RỒI). Do đó:
  - khoảng cách tới đích  = ‖command[:, :2]‖   (KHÔNG trừ root_pos_w nữa)
  - "phía trước" của robot = yaw-frame +Y (index 1) — khớp quy ước vận tốc tiến
    lin_vel_y của locomotion (robot xoay 90° quanh X nên forward = body-Y).
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_dist_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Khoảng cách robot→đích (m) = norm của vector đích trong body yaw-frame."""
    command = env.command_manager.get_command(command_name)   # (N, 3) body-frame
    return torch.linalg.norm(command[:, :2], dim=-1)           # (N,)


def progress_toward_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Thưởng phần khoảng cách rút ngắn mỗi bước (delta distance)."""
    dist = _goal_dist_b(env, command_name)

    if not hasattr(env, "_nav_prev_dist"):
        env._nav_prev_dist = dist.clone()

    # Reset prev_dist ở bước đầu episode để tránh spike khi spawn/đổi đích.
    if hasattr(env, "episode_length_buf"):
        first = env.episode_length_buf == 0
        env._nav_prev_dist[first] = dist[first]

    delta = env._nav_prev_dist - dist            # dương = lại gần đích
    env._nav_prev_dist = dist.clone()
    return delta


def goal_reached(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
                 threshold: float = 0.3) -> torch.Tensor:
    """Bonus nhị phân khi vào trong bán kính `threshold` quanh đích."""
    return (_goal_dist_b(env, command_name) < threshold).float()


def goal_distance_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
                      std: float = 1.0) -> torch.Tensor:
    """Thưởng Gaussian theo khoảng cách tới đích (đỉnh tại đích)."""
    return torch.exp(-_goal_dist_b(env, command_name) / std)


def heading_toward_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
                        v_ref: float = 0.3) -> torch.Tensor:
    """Thưởng khi đích nằm phía TRƯỚC robot VÀ robot đang TIẾN về phía đó.

    = cosine(hướng tiến, hướng đích) × cổng-vận-tốc-tiến.

    CHỐNG FARM ĐỨNG YÊN: nhân thêm cổng theo vận tốc tiến (body +Y). Đứng yên
    (v_fwd≈0) → cổng=0 → heading=0, không thể farm bằng cách chỉ quay mặt về đích.
    Cổng đạt 1 khi v_fwd ≥ v_ref (0.3 m/s); v_fwd≤0 (đứng/lùi) → 0.
    Kết hợp `progress` → robot BẮT BUỘC phải lái tới đích mới có điểm.
    """
    command = env.command_manager.get_command(command_name)         # (N, 3) body-frame
    dist = torch.linalg.norm(command[:, :2], dim=-1).clamp(min=1e-3)
    cos = (command[:, 1] / dist).clamp(-1.0, 1.0)

    asset = env.scene[asset_cfg.name]
    v_fwd = asset.data.root_lin_vel_b[:, 1]                          # body +Y = trục tiến
    gate = (v_fwd / v_ref).clamp(0.0, 1.0)
    return cos * gate
