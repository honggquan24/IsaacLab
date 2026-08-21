# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Con lắc đơn trên xe đẩy (CAD Onshape) — giữ thăng bằng.

Task:
    Isaac-Cart-Pendulum — xe đẩy trên ray, khớp con lắc thụ động.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum --num_envs 4
"""

import gymnasium as gym

from . import agents
from .cart_pendulum_cfg import *  # noqa: F403
from .cart_pendulum_env_cfg import *  # noqa: F403

gym.register(
    id="Isaac-Cart-Pendulum",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_env_cfg:CartPendulumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumPPORunnerCfg",
    },
)
