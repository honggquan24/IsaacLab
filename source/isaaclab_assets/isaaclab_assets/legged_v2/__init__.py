import gymnasium as gym
from . import config
from .config import *

# Nho vao source/isaaclab_assets/isaaclab_assets/__init__.py
# Them dong from .legged_v2 import *

# Train mode headless
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Navigation-Flat-Anymal-C-v0 \

# --task=Isaac-Legged-Robot-V2-Balance \
# --num_envs 1000 \
# --headless --rendering_mode performance

# Train mode debug
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --num_envs 3 \
# --task=Isaac-Legged-Robot-V2-Balance \
# --rendering_mode performance
    
# Continue train
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Legged-Robot-V2-Balance \
# --num_envs 2048 \
# --resume --load_run=2025-12-03_09-35-52 \
# --checkpoint=model_600.pt \
# --video --rendering_mode performance


# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Legged-Robot-V2-Balance \
# --num_envs 4 \
# 'agent.load_run=2025-12-02_10-05-04' \
# 'agent.load_checkpoint="model_300.pt"'

# ./isaaclab.sh -p ./source/isaaclab_assets/isaaclab_assets/legged_v2/tests/run_robot_rl_env.py --device cpu

# ./isaaclab.sh -p -m tensorboard.main --logdir logs

gym.register(
    id="Isaac-Legged-Robot-V2-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.env.legged_v2_env_cfg_balance:LeggedRobotV2EnvCfgBalance",
        "rsl_rl_cfg_entry_point": f"{config.agents.__name__}.rsl_rl_ppo_cfg:LegV2PPORunnerCfgBalance"
    },
)