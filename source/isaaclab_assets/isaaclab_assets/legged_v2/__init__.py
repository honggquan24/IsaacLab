from .legged_v2_cfg import *
from .legged_v2_cfg import *
from .legged_v2_env_cfg import * 
from .legged_v2_env_cfg import *

import gymnasium as gym
from . import agents

# Nho vao source/isaaclab_assets/isaaclab_assets/__init__.py
# Them dong from .legged_v1 import *

# Train mode headless
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Legged-Robot-V2-Pose \
# --num_envs 3 \
# --headless --rendering_mode performance

# Train mode debug
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --num_envs 3 \
# --task=Isaac-Legged-Robot-V2-Pose \
# --rendering_mode performance
    
# Continue train
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Legged-Robot-V2-Pose \
# --num_envs 3 \
# --resume --load_run=2025-12-03_00-37-52 \
# --checkpoint=model_450.pt \
# --video --rendering_mode performance


# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Legged-Robot-V2-Pose \
# --num_envs 4 \
# 'agent.load_run=2025-12-03_00-37-52' \
# 'agent.load_checkpoint="model_450.pt"'


gym.register(
    id="Isaac-Legged-Robot-V2-Pose",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legged_v2_env_cfg:LeggedRobotV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LegV2PPORunnerCfg"
    },
)