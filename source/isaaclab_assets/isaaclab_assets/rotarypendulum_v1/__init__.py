from .rotarypen_cfg import *
from .rotarypen_env_cfg import *

import gymnasium as gym
from . import agents


gym.register(
    id="Isaac-Rotary-V1-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotarypen_env_cfg:RotarypenEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rotaryv1PPORunnerCfg"
    }
)



# Play:
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Rotary-V1-Run \
# --num_envs 16 \
# 'agent.load_run=rotary_pendulum_model' \
# 'agent.load_checkpoint="model_399.pt"'