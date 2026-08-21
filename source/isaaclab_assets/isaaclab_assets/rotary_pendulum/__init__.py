# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Con lắc ngược quay (Furuta) — swing-up rồi giữ thăng bằng.

Chi tiết curriculum: ``docs/ute/rotary_pendulum_curriculum.md``.


Cách đọc lệnh quay video
------------------------
``--video_length`` đếm theo BƯỚC ĐIỀU KHIỂN, không phải giây. Tần số điều khiển
= 1 / (sim.dt × decimation), ghi kèm ở từng task bên dưới.
Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``.
``--load_run`` lấy checkpoint mới nhất trong thư mục run đó; muốn chỉ đúng một
checkpoint thì thay bằng ``--checkpoint <đường/dẫn/model_xxx.pt>``.
Bỏ ``--headless`` nếu muốn xem cửa sổ Isaac Sim trong lúc ghi.

Isaac-Rotary-Pendulum-Balance — một giai đoạn: swing-up + giữ thăng bằng
    60 Hz (sim.dt 1/60, decimation 1) → 60 s = 3600 step; mỗi episode 10 s = 600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Rotary-Pendulum-Balance --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Rotary-Pendulum-Balance --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Rotary-Pendulum-Balance-Stage1 — curriculum giai đoạn 1: chỉ swing-up
    60 Hz → 60 s = 3600 step; mỗi episode 20 s = 1200 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Rotary-Pendulum-Balance-Stage1 --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Rotary-Pendulum-Balance-Stage1 --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Rotary-Pendulum-Balance-Stage2 — curriculum giai đoạn 2: bám vị trí cánh tay
    60 Hz → 60 s = 3600 step; mỗi episode 15 s = 900 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Rotary-Pendulum-Balance-Stage2 --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Rotary-Pendulum-Balance-Stage2 --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>
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
