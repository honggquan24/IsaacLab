"""Joint-unlock auto-curriculum for legged_v3 single-run training.

Curriculum stages (joint unlocked progressively during one training run):
  Phase 0  (iter    0 –  999): Wheels only. Knee/thigh/hip frozen (stiffness=5000, scale=0).
  Phase 1  (iter 1000 – 1999): + Knee   unlocked (stiffness→20, action scale→1).
  Phase 2  (iter 2000 – 2999): + Thigh  unlocked.
  Phase 3  (iter 3000+):       + Hip    unlocked (full 8-DOF control).

Mechanism:
  - All joints are in the action space from the start (fixed NN size).
  - Frozen joints use ImplicitActuatorCfg(stiffness=5000) + JointPositionActionCfg(scale=0).
    → PD target = init_pos every step, large torque holds the joint in place.
  - At each threshold: write_joint_stiffness_to_sim() restores normal stiffness,
    and the corresponding action term's _scale is set to 1.0 to enable control.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from ..legged_v3_curr_cfg import NORMAL_STIFFNESS, NORMAL_DAMPING

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds in env.common_step_counter units (one count = one env.step() call).
# RSL-RL uses num_steps_per_env=24, so one iteration = 24 step() calls.
#   Phase 1: iter 1000 → step 24_000
#   Phase 2: iter 2000 → step 48_000
#   Phase 3: iter 3000 → step 72_000
# ──────────────────────────────────────────────────────────────────────────────
_STEPS_PER_ITER    = 24          # must match num_steps_per_env in PPO runner cfg
_KNEE_UNLOCK_ITER  = 1000
_THIGH_UNLOCK_ITER = 2000
_HIP_UNLOCK_ITER   = 3000

_KNEE_UNLOCK_STEP  = _KNEE_UNLOCK_ITER  * _STEPS_PER_ITER   # 24_000
_THIGH_UNLOCK_STEP = _THIGH_UNLOCK_ITER * _STEPS_PER_ITER   # 48_000
_HIP_UNLOCK_STEP   = _HIP_UNLOCK_ITER   * _STEPS_PER_ITER   # 72_000

# Normal stiffness/damping after unlock — imported from legged_v3_curr_cfg to stay in sync
_NORMAL_STIFFNESS = NORMAL_STIFFNESS   # 20.0
_NORMAL_DAMPING   = NORMAL_DAMPING     # 0.0

# Module-level state: which phases have been applied this session
_unlocked: dict[str, bool] = {"knee": False, "thigh": False, "hip": False}


# ──────────────────────────────────────────────────────────────────────────────

def unlock_joint_phases(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> float | None:
    """Progressive joint unlock curriculum term.

    Called by CurriculumManager on every env reset. Checks env.common_step_counter
    and unlocks joint groups when their iteration threshold is reached.

    Returns the current phase (0–3) for TensorBoard logging.
    """
    step = env.common_step_counter
    robot = env.scene["robot"]

    if step >= _KNEE_UNLOCK_STEP and not _unlocked["knee"]:
        _unlock_group(
            env, robot,
            joint_names=["left_knee_joint", "right_knee_joint"],
            action_term_name="knee_pos",
        )
        _unlocked["knee"] = True
        print(f"[Curriculum] step={step} (iter≈{step//_STEPS_PER_ITER}): "
              f"KNEE unlocked → phase 1 (wheels + knee)")

    if step >= _THIGH_UNLOCK_STEP and not _unlocked["thigh"]:
        _unlock_group(
            env, robot,
            joint_names=["left_thigh_joint", "right_thigh_joint"],
            action_term_name="thigh_pos",
        )
        _unlocked["thigh"] = True
        print(f"[Curriculum] step={step} (iter≈{step//_STEPS_PER_ITER}): "
              f"THIGH unlocked → phase 2 (wheels + knee + thigh)")

    if step >= _HIP_UNLOCK_STEP and not _unlocked["hip"]:
        _unlock_group(
            env, robot,
            joint_names=["left_hip_joint", "right_hip_joint"],
            action_term_name="hip_pos",
        )
        _unlocked["hip"] = True
        print(f"[Curriculum] step={step} (iter≈{step//_STEPS_PER_ITER}): "
              f"HIP unlocked → phase 3 (full 8-DOF)")

    phase = sum([_unlocked["knee"], _unlocked["thigh"], _unlocked["hip"]])
    return float(phase)


def _unlock_group(
    env: ManagerBasedRLEnv,
    robot,
    joint_names: list[str],
    action_term_name: str,
) -> None:
    """Restore normal stiffness and enable action scale for a joint group."""
    # 1. Restore stiffness/damping via ImplicitActuatorCfg runtime write
    joint_ids, _ = robot.find_joints(joint_names)
    n = len(joint_ids)
    stiffness = torch.full((env.num_envs, n), _NORMAL_STIFFNESS, device=env.device)
    damping   = torch.full((env.num_envs, n), _NORMAL_DAMPING,   device=env.device)
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids)
    robot.write_joint_damping_to_sim(damping,     joint_ids=joint_ids)

    # 2. Enable the corresponding action term by setting scale = 1.0
    terms = env.action_manager._terms
    if action_term_name in terms:
        term = terms[action_term_name]
        if hasattr(term, "_scale"):
            if isinstance(term._scale, torch.Tensor):
                term._scale.fill_(1.0)
            else:
                term._scale = 1.0
