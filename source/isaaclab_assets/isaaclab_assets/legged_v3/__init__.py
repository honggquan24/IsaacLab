"""Legged Robot V5 — bipedal wheeled robot (5-bar, USD export từ Onshape).

Task:
  - Isaac-Legged-V5-Wheel: Wheeled locomotion với balance

Train (headless):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Legged-V5-Wheel --num_envs 4096 --headless
"""

import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Legged-V5-Wheel",          # bản MIMIC (policy ra 2 hip, mimic auto)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.legged_v5_wheel_env_cfg:LeggedV5WheelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV5WheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Legged-V5-Wheel-NoMimic",  # bản KHÔNG mimic (policy ra cả 4 hip)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.legged_v5_wheel_env_cfg_nomimic:LeggedV5WheelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV5WheelNoMimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Legged-V5-Navigation",     # tầng cao: command pos → goal, low-level pretrained
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.legged_v5_navigation_env_cfg:LeggedV5NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV5NavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Legged-V5-Warehouse-Nav",  # tầng cao + LiDAR: né vật cản trong kho, tới đích
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.legged_v5_warehouse_nav_env_cfg:LeggedV5WarehouseNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV5WarehouseNavPPORunnerCfg",
    },
)

from .locomotion.legged_v5_wheel_env_cfg import LeggedV5WheelEnvCfg
from .navigation import LeggedV5NavigationEnvCfg, LeggedV5WarehouseNavEnvCfg
