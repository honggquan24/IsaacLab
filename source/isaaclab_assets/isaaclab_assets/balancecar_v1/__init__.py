from .cart_v1_cfg import *
from .cart_v1_env_cfg import *



import gymnasium as gym
from . import agents


# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Cartbalance-V1-Run \
# --num_envs 4 \
# 'agent.load_run=balance' \
# 'agent.load_checkpoint="model_120.pt"'


# Continue train
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Cartbalance-V1-Run \
# --num_envs 1 \
# --resume --load_run=balance \
# --checkpoint=model_120.pt \
# --video --rendering_mode performance



gym.register(
    id="Isaac-Cartbalance-V1-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_v1_env_cfg:CartbalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartbalancePPORunnerCfg"
    }
)