from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================
# Helpers (resolve asset + joint ids)
# ============================================================
def _resolve_articulation_and_joint_ids(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Return (articulation, joint_ids) from SceneEntityCfg(name='robot', joint_names=[...])."""
    art = env.scene[asset_cfg.name]

    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if joint_ids is None:
        # IsaacLab articulation usually provides find_joints()
        if hasattr(art, "find_joints"):
            ids, _ = art.find_joints(asset_cfg.joint_names)
            joint_ids = ids
        else:
            raise RuntimeError("Asset does not support find_joints() and asset_cfg.joint_ids is None.")

    if not torch.is_tensor(joint_ids):
        joint_ids = torch.tensor(joint_ids, device=art.data.joint_pos.device, dtype=torch.long)
    return art, joint_ids



def joint_pos_target_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float | torch.Tensor = 0.0,
    use_angle_wrap: bool = True,
) -> torch.Tensor:
    """Cost = sum((q - target)^2) over selected joints. (Lower is better)"""
    art, joint_ids = _resolve_articulation_and_joint_ids(env, asset_cfg)
    q = art.data.joint_pos[:, joint_ids]  # (N, K)

    if not torch.is_tensor(target):
        tgt = torch.tensor(target, device=q.device, dtype=q.dtype)
    else:
        tgt = target.to(device=q.device, dtype=q.dtype)

    if tgt.ndim == 0:
        tgt = tgt.view(1, 1)
    elif tgt.ndim == 1:
        tgt = tgt.view(1, -1)

    err = q - tgt
    if use_angle_wrap:
        err = wrap_to_pi(err)
    return torch.sum(err * err, dim=1)  # (N,)


def joint_vel_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Cost = sum(|qd|) over selected joints. (Lower is better)"""
    art, joint_ids = _resolve_articulation_and_joint_ids(env, asset_cfg)
    qd = art.data.joint_vel[:, joint_ids]  # (N, K)
    return torch.sum(torch.abs(qd), dim=1)


# ============================================================
# (B) Phi-based terms (chuẩn cho double pendulum góc tương đối)
# ============================================================

def _get_phi1_phi2_from_robot(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """phi1 = wrap(theta1), phi2 = wrap(theta1 + theta2) using joint_pos columns [1], [2]."""
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    theta1 = joint_pos[:, 1]
    theta2 = joint_pos[:, 2]
    phi1 = wrap_to_pi(theta1)
    phi2 = wrap_to_pi(theta1 + theta2)
    return phi1, phi2


def phi_upright_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Reward in [0,1]: (cos(phi1)+cos(phi2) + 2)/4. Works for swing-up + balance."""
    phi1, phi2 = _get_phi1_phi2_from_robot(env)
    cos_sum = torch.cos(phi1) + torch.cos(phi2)  # [-2, 2]
    return (cos_sum + 2.0) / 4.0                 # [0, 1]


def phi_balance_bonus(
    env: ManagerBasedRLEnv,
    angle_thresh: float = 0.2,   # rad
    vel_thresh: float = 1.0,     # rad/s
) -> torch.Tensor:
    """Bonus = 1 when both links are near upright AND angular velocities are small."""
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel

    phi1, phi2 = _get_phi1_phi2_from_robot(env)

    cond_angle = (torch.abs(phi1) < angle_thresh) & (torch.abs(phi2) < angle_thresh)
    cond_vel = (torch.abs(joint_vel[:, 1]) < vel_thresh) & (torch.abs(joint_vel[:, 2]) < vel_thresh)
    return (cond_angle & cond_vel).float()


def cart_pos_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float = 0.0,
) -> torch.Tensor:
    """Cost cart position squared (for centering). Use the cart joint only."""
    return joint_pos_target_l2(env, asset_cfg=asset_cfg, target=target, use_angle_wrap=False)


# ============================================================
# RewardCfg (đơn giản, giống mẫu ảnh)
# - Nếu em muốn swing-up + giữ thăng bằng luôn:
#   dùng phi_upright_reward + phi_balance_bonus
# ============================================================

