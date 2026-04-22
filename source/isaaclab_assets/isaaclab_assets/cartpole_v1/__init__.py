from .cartpole_v1_cfg import *
from .cartpole_v1_env_cfg import *

import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Cartpole-V1-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_v1_env_cfg:CartPoleV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPoleV1PPORunnerCfg"
    }
)

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Cartpole-V1-Run \
# --num_envs 4 \
# 'agent.load_run=balance1' \
# 'agent.load_checkpoint="model_60.pt"'