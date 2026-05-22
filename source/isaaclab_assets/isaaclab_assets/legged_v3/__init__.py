"""Legged Robot V3 — 2-legged wheeled robot.

Tasks:
  - Isaac-Legged-V3-Wheel:      Wheeled locomotion with balance
  - Isaac-Legged-V3-Leg:        Leg-based locomotion
  - Isaac-Legged-V3-Curriculum: Auto-curriculum locomotion
  - Isaac-Legged-V3-Navigation: Hierarchical navigation (high-level over pre-trained locomotion)

Train locomotion (headless):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Wheel --num_envs 4096 --headless

Train navigation (requires trained locomotion checkpoint):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Navigation --num_envs 1024 --headless
"""

import gymnasium as gym
from . import agents

# Register tasks first (before env-cfg imports) to avoid circular imports:
# env-cfg modules import isaaclab_tasks.manager_based.* which would re-enter
# this package during isaaclab_tasks initialisation.

gym.register(
    id="Isaac-Legged-V3-Wheel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.legged_v3_wheel_env_cfg:LeggedV3WheelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV3WheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Legged-V3-Leg",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.legged_v3_leg_env_cfg:LeggedV3LegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV3LegPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Legged-V3-Curriculum",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.curriculum.env_cfg:LeggedV3CurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV3CurriculumPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Legged-V3-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.legged_v3_wheel_navigation_env_cfg:LeggedV3WheelNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV3WheelNavPPORunnerCfg",
    },
)

from .legged_v3_cfg import *
from .locomotion.legged_v3_wheel_env_cfg import *
from .locomotion.legged_v3_leg_env_cfg import *
from .locomotion.curriculum import LeggedV3CurriculumEnvCfg
from .navigation import LeggedV3WheelNavigationEnvCfg
