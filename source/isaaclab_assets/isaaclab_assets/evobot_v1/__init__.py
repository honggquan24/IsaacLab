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


# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Balance \
# --num_envs 7000 \
# --resume --load_run=2026-01-12_14-45-12 \
# --checkpoint=model_2000.pt \
# --video --rendering_mode performance --headless


# Play/evaluate trained model
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Evobot-V1-Balance \
# --num_envs 1 \
# 'agent.load_run=2026-01-12_14-45-12' \
# 'agent.load_checkpoint="model_2000.pt"'

# Test environment loading
# ./isaaclab.sh -p ./source/isaaclab_assets/isaaclab_assets/evobot_v1/tests/run_robot_rl_env.py --device cpu

# ./isaaclab.sh -p -m tensorboard.main --logdir logs

##
# Balance Task
##
gym.register(
    id="Isaac-Evobot-V1-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.balance.__name__}.evobot_v1_env_cfg_balance:EvobotV1EnvCfgBalance",
        "rsl_rl_cfg_entry_point": f"{config.balance.agents.__name__}.rsl_rl_ppo_cfg:EvobotPPORunnerCfgBalance"
    },
)

##
# Velocity Balance Task (Balance + Velocity Command Following)
##

# Train velocity balance (balance with velocity commands)
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Velocity-Balance \
# --num_envs 1024 \
# --headless --rendering_mode performance

# Train mode debug
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Velocity-Balance \
# --num_envs 3 \
# --rendering_mode performance

# Continue train
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Velocity-Balance \
# --num_envs 7000 \
# --resume --load_run=2026-01-12_14-45-12 \
# --checkpoint=model_2000.pt \
# --headless --video --rendering_mode performance

# Play/evaluate velocity balance
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Evobot-V1-Velocity-Balance \
# --num_envs 4 \
# 'agent.load_run=<run_name>' \
# 'agent.load_checkpoint="model_500.pt"'

gym.register(
    id="Isaac-Evobot-V1-Velocity-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.navigation.__name__}.evobot_v1_velocity_env_cfg:EvobotV1VelocityBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{config.navigation.agents.__name__}.rsl_rl_ppo_cfg:EvobotVelocityPPORunnerCfg",
    },
)

##
# Navigation Task (Approach 2: Hierarchical - Using pre-trained balance policy)
##

# IMPORTANT: Update policy_path in evobot_v1_navigation_pretrained_env_cfg.py line 99
# before training. Export balance policy first:
# 1. Train balance task: --task=Isaac-Evobot-V1-Balance
# 2. Export policy: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#    --task Isaac-Evobot-V1-Balance --num_envs 1 \
#    'agent.load_run=<balance_run_name>' 'agent.load_checkpoint="model_300.pt"'
# 3. Update policy_path in evobot_v1_navigation_pretrained_env_cfg.py
# 4. Train navigation: --task=Isaac-Evobot-V1-Navigation-Pretrained

# Train hierarchical navigation (requires pre-trained balance policy)
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
# --task=Isaac-Evobot-V1-Navigation-Pretrained \
# --num_envs 512 \
# --headless --rendering_mode performance

# Play hierarchical navigation
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
# --task Isaac-Evobot-V1-Navigation-Pretrained-Play \
# --num_envs 16 \
# 'agent.load_run=<nav_pretrained_run_name>' \
# 'agent.load_checkpoint="model_500.pt"'

gym.register(
    id="Isaac-Evobot-V1-Navigation-Pretrained",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.navigation.__name__}.evobot_v1_navigation_pretrained_env_cfg:EvobotV1NavigationPretrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{config.navigation.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Evobot-V1-Navigation-Pretrained-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.navigation.__name__}.evobot_v1_navigation_pretrained_env_cfg:EvobotV1NavigationPretrainedEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{config.navigation.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)
