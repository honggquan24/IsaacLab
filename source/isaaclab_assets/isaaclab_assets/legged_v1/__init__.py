from .legged_v1_cfg import *
from .legged_v1_env_cfg import * 
import gymnasium as gym
from . import agents

# Nho vao source/isaaclab_assets/isaaclab_assets/__init__.py
# Them dong from .legged_v1 import *
gym.register(
    id="Isaac-Legged-Robot-V1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legged_v1_env_cfg:LeggedRobotV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LegV1PPORunnerCfg"
    },
)