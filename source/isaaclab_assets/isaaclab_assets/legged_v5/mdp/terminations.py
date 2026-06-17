"""Custom termination functions for Legged Robot V3."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation_from_default(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate khi robot lệch quá limit_angle so với orientation mặc định (init_state.rot).

    Khác bad_orientation gốc: đo góc lệch từ default quat thay vì từ world-up.
    Dùng khi robot spawn không ở identity (ví dụ rot=(0.7071,-0.7071,0,0)).

    Args:
        limit_angle: Góc lệch tối đa cho phép (radian). Ví dụ math.pi/3 = 60°.
    """
    asset = env.scene[asset_cfg.name]

    quat_cur = asset.data.root_quat_w          # (N, 4) — (w,x,y,z)
    quat_ref = asset.data.default_root_state[:, 3:7]  # (N, 4) — init_state.rot

    # q_rel = q_ref^{-1} * q_cur  →  góc lệch từ default
    quat_ref_inv = quat_ref * torch.tensor([1, -1, -1, -1], device=quat_ref.device)
    quat_rel = quat_mul(quat_ref_inv, quat_cur)

    # Angular distance = 2 * acos(|w_rel|), range [0, π]
    angle = 2.0 * torch.acos(quat_rel[:, 0].abs().clamp(0.0, 1.0))

    return angle > limit_angle
