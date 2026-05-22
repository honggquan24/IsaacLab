"""Performance-based command-range curriculum for legged_v3 wheel locomotion.

Phases advance when the robot demonstrates mastery of the current task,
measured by an EMA of episode rewards normalized by episode length.

  Phase 0: Stand still + reach height.
           Advance when: EMA(track_base_height_exp / step) > HEIGHT_THRESH
                     AND EMA(upright / step) > UPRIGHT_THRESH  (stable)
  Phase 1: Slow movement ≤ 0.5 m/s.
           Advance when: EMA(track_lin_vel_xy_exp / step) > VEL_THRESH
  Phase 2: Full speed ≤ 1.0 m/s. (final)

EMA smoothing factor α=0.05 → window ≈ 20 episodes per env before stable.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

# ── Thresholds (reward per step, normalized by weight) ────────────────────────
# track_base_height_exp weight=5.0, max reward/step=1.0 → per-step max = 5.0
# Advance phase 0 when robot reaches ≥ 60% of max height reward per step
_HEIGHT_THRESH  = 3.0    # track_base_height_exp per step (weight=5.0 → max=5.0)
_UPRIGHT_THRESH = -1.5   # upright per step (weight=-30 → worse = more negative)

# track_lin_vel_xy_exp weight=3.0, max=1.0/step → max=3.0
# Advance phase 1 when robot reaches ≥ 60% of max velocity reward per step
_VEL_THRESH     = 1.8    # track_lin_vel_xy_exp per step (weight=3.0 → max=3.0)

# Minimum episodes per env before allowing phase advance (avoid premature jump)
_MIN_EPISODES = 50

# EMA smoothing factor: α=0.05 → ~20-episode window
_EMA_ALPHA = 0.05

# ── State ─────────────────────────────────────────────────────────────────────
_phase = 0
_episode_count = 0

# EMA of reward-per-step for each tracked term (scalar, averaged across envs)
_ema_height  = 0.0
_ema_upright = 0.0
_ema_vel_xy  = 0.0

_phase_logged: dict[int, bool] = {1: False, 2: False}


def expand_velocity_command_range(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> float:
    """Advance velocity command range when performance threshold is met.

    Called by CurriculumManager on every env reset (env_ids = just-reset envs).
    Returns current phase (0/1/2) for TensorBoard logging.
    """
    global _phase, _episode_count, _ema_height, _ema_upright, _ema_vel_xy

    if len(env_ids) == 0:
        return float(_phase)

    # ── Compute reward-per-step for just-terminated envs ─────────────────────
    ep_len = env.episode_length_buf[env_ids].float().clamp(min=1)
    ep_sums = env.reward_manager._episode_sums

    height_per_step  = (ep_sums["track_base_height_exp"][env_ids] / ep_len).mean().item()
    upright_per_step = (ep_sums["upright"][env_ids]                / ep_len).mean().item()
    vel_per_step     = (ep_sums["track_lin_vel_xy_exp"][env_ids]   / ep_len).mean().item()

    # ── Update EMA ────────────────────────────────────────────────────────────
    _ema_height  = (1 - _EMA_ALPHA) * _ema_height  + _EMA_ALPHA * height_per_step
    _ema_upright = (1 - _EMA_ALPHA) * _ema_upright + _EMA_ALPHA * upright_per_step
    _ema_vel_xy  = (1 - _EMA_ALPHA) * _ema_vel_xy  + _EMA_ALPHA * vel_per_step

    _episode_count += len(env_ids)

    # ── Phase transitions ─────────────────────────────────────────────────────
    if _phase == 0 and _episode_count >= _MIN_EPISODES * env.num_envs:
        if _ema_height >= _HEIGHT_THRESH and _ema_upright >= _UPRIGHT_THRESH:
            _set_phase(env, 1)

    elif _phase == 1 and _episode_count >= _MIN_EPISODES * env.num_envs:
        if _ema_vel_xy >= _VEL_THRESH:
            _set_phase(env, 2)

    return float(_phase)


def _set_phase(env: ManagerBasedRLEnv, new_phase: int) -> None:
    global _phase, _episode_count

    cmd_term = env.command_manager._terms["velocity_command"]
    ranges   = cmd_term.cfg.ranges

    if new_phase == 1 and not _phase_logged[1]:
        ranges.lin_vel_x = (-0.5, 0.5)
        ranges.ang_vel_z = (-0.5, 0.5)
        cmd_term.cfg.rel_standing_envs = 0.3
        print(f"[Curriculum] Phase 0→1: "
              f"height_EMA={_ema_height:.2f} upright_EMA={_ema_upright:.2f} "
              f"→ slow movement ≤ 0.5 m/s")
        _phase_logged[1] = True

    elif new_phase == 2 and not _phase_logged[2]:
        ranges.lin_vel_x = (-1.0, 1.0)
        ranges.ang_vel_z = (-1.0, 1.0)
        cmd_term.cfg.rel_standing_envs = 0.1
        print(f"[Curriculum] Phase 1→2: "
              f"vel_EMA={_ema_vel_xy:.2f} "
              f"→ full speed ≤ 1.0 m/s")
        _phase_logged[2] = True

    _phase = new_phase
    _episode_count = 0  # reset counter for next phase
