from .cartpole_v2_cfg import *
from .cartpole_v2_env_cfg import *

import gymnasium as gym
from . import agents

# Nho vao source/isaaclab_assets/isaaclab_assets/__init__.py
# Them dong from .legged_v1 import *

# Train mode headless
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Legged-Robot-V2-Pose \
# --num_envs 4 \
# --headless --rendering_mode performance

# Train mode debug
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --num_envs 10000 \
# --task=Isaac-Cartpole-V2-Run \
# --rendering_mode performance
    
# Continue train
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Cartpole-V2-Run \
# --num_envs 10000 \
# --resume --load_run=balance \
# --checkpoint=model_50.pt \
# --video --rendering_mode performance


# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Cartpole-V2-Run \
# --num_envs 4 \
# 'agent.load_run=2025-12-05_09-26-26' \
# 'agent.load_checkpoint="model_250.pt"'

# ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --device cpu

# ./isaaclab.sh -p -m tensorboard.main --logdir=logs

gym.register(
    id="Isaac-Cartpole-V2-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_v2_env_cfg:CartPoleV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPoleV2PPORunnerCfg"
    }
)