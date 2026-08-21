# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Con lắc kép trên xe đẩy — swing-up và giữ thăng bằng.

Task:
    Isaac-Cart-Pendulum-Double — xe đẩy điều khiển bằng lực, hai khớp con lắc thụ động.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum-Double --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum-Double --num_envs 4
"""

import gymnasium as gym

from . import agents
from .cart_pendulum_double_cfg import CART_PENDULUM_DOUBLE_CFG  # noqa: F401
from .cart_pendulum_double_env_cfg import *  # noqa: F403

gym.register(
    id="Isaac-Cart-Pendulum-Double",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_double_env_cfg:CartPendulumDoubleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumDoublePPORunnerCfg",
    },
)
