# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Xe hai bánh tự cân bằng — giữ thăng bằng và điều hướng tới đích.

Cách đọc lệnh quay video
------------------------
``--video_length`` đếm theo BƯỚC ĐIỀU KHIỂN, không phải giây. Tần số điều khiển
= 1 / (sim.dt × decimation), ghi kèm ở từng task bên dưới.
Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``.
``--load_run`` lấy checkpoint mới nhất trong thư mục run đó; muốn chỉ đúng một
checkpoint thì thay bằng ``--checkpoint <đường/dẫn/model_xxx.pt>``.
Bỏ ``--headless`` nếu muốn xem cửa sổ Isaac Sim trong lúc ghi.

Isaac-Balance-Car — giữ thăng bằng, bám lệnh vận tốc
    30 Hz (sim.dt 1/60, decimation 2) → 60 s = 1800 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Balance-Car --num_envs 2048 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Balance-Car --num_envs 4 --headless \
        --video --video_length 1800 --load_run <tên_run>

Isaac-Balance-Car-Navigation — tầng cao tới đích, học từ đầu
    30 Hz → 60 s = 1800 step; mỗi episode 5 s = 150 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Balance-Car-Navigation --num_envs 2048 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Balance-Car-Navigation-Play --num_envs 16 --headless \
        --video --video_length 1800 --load_run <tên_run>

Isaac-Balance-Car-Navigation-Pretrained — tầng cao dùng policy thăng bằng đã train
    6 Hz (decimation 2×5) → 60 s = 360 step; mỗi episode 5 s = 30 step
    Phải train ``Isaac-Balance-Car`` trước, rồi sửa ``policy_path`` trong
    ``navigation/navigation_pretrained_env_cfg.py`` trỏ tới ``exported/policy.pt``.

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Balance-Car-Navigation-Pretrained --num_envs 2048 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Balance-Car-Navigation-Pretrained-Play --num_envs 16 --headless \
        --video --video_length 360 --load_run <tên_run>
"""

import gymnasium as gym

from . import agents
from .balance_car_cfg import *
from .balance_env_cfg import *
from .navigation import agents as nav_agents

gym.register(
    id="Isaac-Balance-Car",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:BalanceCarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceCarPPORunnerCfg",
    },
)

# ==============================================================================
# Register Navigation Environment
# ==============================================================================
gym.register(
    id="Isaac-Balance-Car-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_env_cfg:BalanceCarNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Balance-Car-Navigation-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_env_cfg:BalanceCarNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPPORunnerCfg",
    },
)

# ==============================================================================
# Register Navigation with Pre-trained Balance Policy Environment
# ==============================================================================
# Train navigation with pre-trained balance policy:

# Play navigation with pre-trained policy:

gym.register(
    id="Isaac-Balance-Car-Navigation-Pretrained",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_pretrained_env_cfg:BalanceCarNavigationPretrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Balance-Car-Navigation-Pretrained-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_pretrained_env_cfg:BalanceCarNavigationPretrainedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPPORunnerCfg",
    },
)
