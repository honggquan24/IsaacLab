"""Performance-based command-range curriculum for legged_v3 wheel locomotion.

Phases advance when the robot demonstrates mastery of the current task,
measured by an EMA of episode rewards normalized by episode length.

  Phase 0 → 1: height EMA >= MASTERY * W_HEIGHT  AND  upright EMA >= (1-MASTERY) * W_UPRIGHT
  Phase 1 → 2: vel_xy EMA >= MASTERY * W_VEL_XY  AND  vel_yaw EMA >= MASTERY * W_VEL_YAW

To change mastery level or reward weights: edit only the constants below.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

# ── Mastery ratio ─────────────────────────────────────────────────────────────
# Fraction of max per-step reward that must be sustained before phase advance.
_MASTERY = 0.80

# ── Reward weights (must mirror RewardCfg in legged_v3_wheel_env_cfg.py) ──────
# _W_HEIGHT  =  8.0  # disabled: height reward off
_W_UPRIGHT = -5.0   # upright weight  (negative → lower is worse)
_W_VEL_XY  =  5.0   # track_lin_vel_xy_exp weight
_W_VEL_YAW =  4.0   # track_ang_vel_z_exp weight

# ── Thresholds derived from weights and mastery ───────────────────────────────
# Positive rewards: threshold = MASTERY × weight
# _HEIGHT_THRESH  = _MASTERY * _W_HEIGHT         # disabled: height reward off
_VEL_XY_THRESH  = _MASTERY * _W_VEL_XY          # e.g. 0.8 × 5.0 = 4.0
_VEL_YAW_THRESH = _MASTERY * _W_VEL_YAW         # e.g. 0.8 × 4.0 = 3.2

# Negative reward (upright): best=0, worst=W_UPRIGHT.
# 80% mastery → allow only (1-MASTERY) fraction of worst-case error.
_UPRIGHT_THRESH = (1.0 - _MASTERY) * _W_UPRIGHT  # e.g. 0.2 × -5.0 = -1.0

# EMA smoothing factor: α=0.03 → ~33-episode window for stable signal
_EMA_ALPHA = 0.03

# ── State ─────────────────────────────────────────────────────────────────────
_phase = 0

_ema_height  = 0.0
_ema_upright = 0.0
_ema_vel_xy  = 0.0
_ema_vel_yaw = 0.0

_phase_logged: dict[int, bool] = {1: False, 2: False}


def expand_velocity_command_range(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> float:
    """Advance velocity command range when 80% reward mastery is sustained.

    Called by CurriculumManager on every env reset (env_ids = just-reset envs).
    Returns current phase (0/1/2) for TensorBoard logging.
    """
    global _phase, _ema_height, _ema_upright, _ema_vel_xy, _ema_vel_yaw

    if len(env_ids) == 0:
        return float(_phase)

    # ── Compute reward-per-step for just-terminated envs ─────────────────────
    ep_len = env.episode_length_buf[env_ids].float().clamp(min=1)
    ep_sums = env.reward_manager._episode_sums

    # height disabled → skip ep_sums["track_base_height_exp"]
    upright_per_step = (ep_sums["upright"][env_ids]                / ep_len).mean().item()
    vel_xy_per_step  = (ep_sums["track_lin_vel_xy_exp"][env_ids]   / ep_len).mean().item()
    vel_yaw_per_step = (ep_sums["track_ang_vel_z_exp"][env_ids]    / ep_len).mean().item()

    # ── Update EMA ────────────────────────────────────────────────────────────
    _ema_upright = (1 - _EMA_ALPHA) * _ema_upright + _EMA_ALPHA * upright_per_step
    _ema_vel_xy  = (1 - _EMA_ALPHA) * _ema_vel_xy  + _EMA_ALPHA * vel_xy_per_step
    _ema_vel_yaw = (1 - _EMA_ALPHA) * _ema_vel_yaw + _EMA_ALPHA * vel_yaw_per_step

    # ── Phase transitions ─────────────────────────────────────────────────────
    if _phase == 0:
        if _ema_upright >= _UPRIGHT_THRESH:
            _set_phase(env, 1)

    elif _phase == 1:
        if _ema_vel_xy >= _VEL_XY_THRESH and _ema_vel_yaw >= _VEL_YAW_THRESH:
            _set_phase(env, 2)

    return float(_phase)


def _set_phase(env: ManagerBasedRLEnv, new_phase: int) -> None:
    global _phase

    cmd_term = env.command_manager._terms["velocity_command"]
    ranges   = cmd_term.cfg.ranges

    if new_phase == 1 and not _phase_logged[1]:
        ranges.lin_vel_x = (-0.5, 0.5)
        ranges.ang_vel_z = (-0.5, 0.5)
        cmd_term.cfg.rel_standing_envs = 0.3
        print(
            f"[Curriculum] Phase 0→1: "
            f"upright_EMA={_ema_upright:.2f}/{_UPRIGHT_THRESH} "
            f"→ slow movement ≤ 0.5 m/s"
        )
        _phase_logged[1] = True

    elif new_phase == 2 and not _phase_logged[2]:
        ranges.lin_vel_x = (-1.0, 1.0)
        ranges.ang_vel_z = (-1.0, 1.0)
        cmd_term.cfg.rel_standing_envs = 0.1
        print(
            f"[Curriculum] Phase 1→2: "
            f"vel_xy_EMA={_ema_vel_xy:.2f}/{_VEL_XY_THRESH} "
            f"vel_yaw_EMA={_ema_vel_yaw:.2f}/{_VEL_YAW_THRESH} "
            f"→ full speed ≤ 1.0 m/s"
        )
        _phase_logged[2] = True

    _phase = new_phase
