# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot bipedal wheel (V5) — hai bánh, chân 5 khâu (5-bar, USD export từ Onshape).

Task:
  - Isaac-Wheeled-Biped-Wheel: Wheeled locomotion với balance

Train (headless):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Wheeled-Biped-Wheel --num_envs 4096 --headless
"""

import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Wheeled-Biped-Wheel",          # bản MIMIC (policy ra 2 hip, mimic auto)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_env_cfg:WheeledBipedWheelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Wheel-Play",     # như bản MIMIC nhưng ít env + camera bám (quay video)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_env_cfg:WheeledBipedWheelPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Wheel-NoMimic",  # bản KHÔNG mimic (policy ra cả 4 hip)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_nomimic_env_cfg:WheeledBipedWheelNoMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelNoMimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Wheel-PIANN",    # bánh = PI-ANN (mạng xuất Kp/Ki/Kd, PID quy ra lệnh bánh)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_piann_env_cfg:WheeledBipedWheelPIANNEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelPIANNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Navigation",     # tầng cao: command pos → goal, low-level pretrained
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_env_cfg:WheeledBipedNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Warehouse-Nav",  # tầng cao + LiDAR: né vật cản trong kho, tới đích
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.warehouse_nav_env_cfg:WheeledBipedWarehouseNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Obstacle-Nav",   # tầng cao + LiDAR: né vật cản trên terrain sinh thủ tục (train song song)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.obstacle_nav_env_cfg:WheeledBipedObstacleNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedObstacleNavPPORunnerCfg",
    },
)

# Re-export ở cuối file (sau các lần đăng ký) để lỗi import của một env cfg không làm
# hỏng việc đăng ký các task còn lại.
from .locomotion.wheel_env_cfg import WheeledBipedWheelEnvCfg, WheeledBipedWheelPlayEnvCfg  # noqa: E402
from .navigation import WheeledBipedNavigationEnvCfg, WheeledBipedWarehouseNavEnvCfg  # noqa: E402
