# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Xe hai bánh tự cân bằng — giữ thăng bằng và điều hướng tới đích.

Task:
    Isaac-Balance-Car                            giữ thăng bằng, bám lệnh vận tốc
    Isaac-Balance-Car-Navigation                 tầng cao tới đích, học từ đầu
    Isaac-Balance-Car-Navigation-Play            như trên, ít env
    Isaac-Balance-Car-Navigation-Pretrained      tầng cao dùng policy thăng bằng đã train
    Isaac-Balance-Car-Navigation-Pretrained-Play như trên, ít env

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Balance-Car --num_envs 2048 --headless
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
