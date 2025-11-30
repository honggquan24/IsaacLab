from .legged_v2_cfg import *
from .legged_v2_cfg_test import *
from .legged_v2_env_cfg import * 
from .legged_v2_env_cfg_test import *

import gymnasium as gym
from . import agents

# Nho vao source/isaaclab_assets/isaaclab_assets/__init__.py
# Them dong from .legged_v1 import *
gym.register(
    id="Isaac-Legged-Robot-V2-Pose",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legged_v2_env_cfg_test:LeggedRobotV2EnvCfgTest",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LegV2PPORunnerCfg"
    },
)