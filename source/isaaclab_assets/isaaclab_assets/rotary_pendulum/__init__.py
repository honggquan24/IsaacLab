# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Con lắc ngược quay (Furuta) — swing-up rồi giữ thăng bằng.

Task:
    Isaac-Rotary-Pendulum-Balance         một giai đoạn: swing-up + giữ thăng bằng
    Isaac-Rotary-Pendulum-Balance-Stage1  curriculum giai đoạn 1: chỉ swing-up
    Isaac-Rotary-Pendulum-Balance-Stage2  curriculum giai đoạn 2: bám vị trí cánh tay

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Rotary-Pendulum-Balance --num_envs 4096 --headless

Chi tiết curriculum: ``docs/ute/rotary_pendulum_curriculum.md``.
"""

import gymnasium as gym

from . import balance

gym.register(
    id="Isaac-Rotary-Pendulum-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{balance.__name__}.balance_env_cfg:RotaryPendulumBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{balance.agents.__name__}.rsl_rl_ppo_cfg:RotaryPendulumBalancePPORunnerCfg",
    },
)

# ============================================================================
# CURRICULUM LEARNING - Stage 1: Swing-up Only
# ============================================================================
gym.register(
    id="Isaac-Rotary-Pendulum-Balance-Stage1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{balance.__name__}.balance_curriculum:RotaryPendulumBalanceStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{balance.agents.__name__}.rsl_rl_ppo_curriculum_cfg:RotaryPendulumStage1PPORunnerCfg",
    },
)

# ============================================================================
# CURRICULUM LEARNING - Stage 2: Balance + Heading Tracking
# ============================================================================
gym.register(
    id="Isaac-Rotary-Pendulum-Balance-Stage2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{balance.__name__}.balance_curriculum:RotaryPendulumBalanceStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{balance.agents.__name__}.rsl_rl_ppo_curriculum_cfg:RotaryPendulumStage2PPORunnerCfg",
    },
)
