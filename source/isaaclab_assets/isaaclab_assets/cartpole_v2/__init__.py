from .cartpole_v2_cfg import *
from .cartpole_v2_env_cfg import *

import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Cartpole-V2-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_v2_env_cfg:CartPoleV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPoleV2PPORunnerCfg"
    }
)