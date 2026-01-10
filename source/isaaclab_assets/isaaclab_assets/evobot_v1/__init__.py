import gymnasium as gym
from . import config
from .config import *

# Remember to add to source/isaaclab_assets/isaaclab_assets/__init__.py:
# from .evobot_v1 import *

# Train mode headless
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Balance \
# --num_envs 1024 \
# --headless --rendering_mode performance

# Train mode debug
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --num_envs 3 \
# --task=Isaac-Evobot-V1-Balance \
# --rendering_mode performance

# Continue train
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Balance \
# --num_envs 1024 \
# --resume --load_run=<run_name> \
# --checkpoint=model_<num>.pt \
# --video --rendering_mode performance

# Play/evaluate trained model
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Evobot-V1-Balance \
# --num_envs 4 \
# 'agent.load_run=<run_name>' \
# 'agent.load_checkpoint="model_<num>.pt"'

# Test environment loading
# ./isaaclab.sh -p ./source/isaaclab_assets/isaaclab_assets/evobot_v1/tests/run_robot_rl_env.py

# ./isaaclab.sh -p -m tensorboard.main --logdir logs

##
# Balance Task
##
gym.register(
    id="Isaac-Evobot-V1-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.__name__}.env.evobot_v1_env_cfg_balance:EvobotV1EnvCfgBalance",
        "rsl_rl_cfg_entry_point": f"{config.agents.__name__}.rsl_rl_ppo_cfg:EvobotPPORunnerCfgBalance"
    },
)

##
# Navigation Task
##

# Train navigation
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Navigation \
# --num_envs 1024 \
# --headless --rendering_mode performance

# Play navigation
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Evobot-V1-Navigation-Play \
# --num_envs 16 \
# 'agent.load_run=evobot_v1_navigation' \
# 'agent.load_checkpoint="model_500.pt"'

gym.register(
    id="Isaac-Evobot-V1-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.navigation.__name__}:EvobotV1NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{config.navigation.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Evobot-V1-Navigation-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.navigation.__name__}:EvobotV1NavigationEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{config.navigation.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)