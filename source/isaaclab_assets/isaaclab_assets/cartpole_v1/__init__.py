from .cartpole_v1_cfg import *
from .cartpole_v1_env_cfg import *

import gymnasium as gym
from . import agents

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-V1-Balance --num_envs 4096 --resume --load_run=pose_1 --checkpoint=model_150.pt --video

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#   --task Isaac-Cartpole-V1-Balance \
#   --num_envs 4 \
#   'agent.load_run=complete' \
#   'agent.load_checkpoint="model_50.pt"' \
#   --device cpu

gym.register(
    id="Isaac-Cartpole-V1-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_v1_env_cfg:CartPoleV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPoleV1PPORunnerCfg"
    }
)
