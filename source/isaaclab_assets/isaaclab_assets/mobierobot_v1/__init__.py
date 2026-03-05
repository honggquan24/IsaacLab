from .mobierobotv1_cfg import *
from .mobierobot_env_cfg import *


import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-MobieRobot-V1-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mobierobot_env_cfg:MobieRobotV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MobieRobotPPORunnerCfg"
    }
)

