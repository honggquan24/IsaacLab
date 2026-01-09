from .cart_v1_cfg import *
from .cart_v1_env_cfg import *


import gymnasium as gym
from . import agents
from .config.navigation import agents as nav_agents


# ==============================================================================
# Balance Task Commands
# ==============================================================================
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Cartbalance-V1-Run \
# --num_envs 4 \
# 'agent.load_run=timestamp' \
# 'agent.load_checkpoint="model_399.pt"'

# Continue train
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Cartbalance-V1-Run \
# --num_envs 1 \
# --resume --load_run=balance \
# --checkpoint=model_120.pt \
# --video --rendering_mode performance

# ==============================================================================
# Navigation Task Commands
# ==============================================================================
# Train navigation:
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task Isaac-Cartbalance-V1-Navigation \
# --num_envs 2048 \
# --headless --rendering_mode performance

# Play navigation:
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Cartbalance-V1-Navigation-Play \
# --num_envs 16 \
# 'agent.load_run=cart_v1_navigation' \
# 'agent.load_checkpoint="model_500.pt"'


# ==============================================================================
# Register Balance Environment
# ==============================================================================
gym.register(
    id="Isaac-Cartbalance-V1-Run",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_v1_env_cfg:CartbalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartbalancePPORunnerCfg"
    }
)

# ==============================================================================
# Register Navigation Environment
# ==============================================================================
gym.register(
    id="Isaac-Cartbalance-V1-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.navigation.cart_v1_navigation_env_cfg:CartV1NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:CartV1NavigationPPORunnerCfg"
    }
)

gym.register(
    id="Isaac-Cartbalance-V1-Navigation-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.navigation.cart_v1_navigation_env_cfg:CartV1NavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:CartV1NavigationPPORunnerCfg"
    }
)

# ==============================================================================
# Register Navigation with Pre-trained Balance Policy Environment
# ==============================================================================
# Train navigation with pre-trained balance policy:
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task Isaac-Cartbalance-V1-Navigation-Pretrained \
# --num_envs 6000 \
# --rendering_mode performance

# Play navigation with pre-trained policy:
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Cartbalance-V1-Navigation-Pretrained-Play \
# --num_envs 16 \
# 'agent.load_run=cart_v1_navigation_pretrained' \
# 'agent.load_checkpoint="model_500.pt"'

gym.register(
    id="Isaac-Cartbalance-V1-Navigation-Pretrained",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.navigation.cart_v1_navigation_pretrained_env_cfg:CartV1NavigationPretrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:CartV1NavigationPPORunnerCfg"
    }
)

gym.register(
    id="Isaac-Cartbalance-V1-Navigation-Pretrained-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.navigation.cart_v1_navigation_pretrained_env_cfg:CartV1NavigationPretrainedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:CartV1NavigationPPORunnerCfg"
    }
)