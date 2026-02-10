import gymnasium as gym
from . import navigation

# ============================================================================
# ROTARY PENDULUM V2 (Furuta Pendulum) - Isaac Lab Training & Evaluation
# ============================================================================
#
# This file registers all Rotary Pendulum V2 tasks for Isaac Lab.
#
# IMPORTANT: Add to source/isaaclab_assets/isaaclab_assets/__init__.py:
#   from .rotary_pendulum_v2 import *
#
# ============================================================================
# QUICK START - Training Commands
# ============================================================================
#
# 1. SWING-UP / BALANCE TASK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Train (production):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-RotaryPendulum-V2-Balance \
#       --num_envs 4096 \
#       --headless --rendering_mode performance
#
#   Train (debug with 3 envs):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-RotaryPendulum-V2-Balance \
#       --num_envs 3 \
#       --rendering_mode performance
#
#   Continue training from checkpoint:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-RotaryPendulum-V2-Balance \
#       --num_envs 4096 \
#       --resume --load_run=<run_name> \
#       --checkpoint=model_<num>.pt \
#       --video --rendering_mode performance \
#       --headless
#
#   Evaluate (test trained model):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-RotaryPendulum-V2-Balance \
#       --num_envs 4 \
#       'agent.load_run=<run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#
# ============================================================================
# USEFUL UTILITIES
# ============================================================================
#
#   List all registered environments:
#   ./isaaclab.sh -p scripts/environments/list_envs.py | grep RotaryPendulum
#
#   View TensorBoard training logs:
#   ./isaaclab.sh -p -m tensorboard.main --logdir logs
#
# ============================================================================

# ============================================================================
# ENVIRONMENT REGISTRATIONS
# ============================================================================

# Swing-up / Balance Task: Isaac-RotaryPendulum-V2-Balance
gym.register(
    id="Isaac-RotaryPendulum-V2-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.balance.__name__}.rotary_pendulum_v2_balance_env_cfg:RotaryPendulumV2BalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.balance.agents.__name__}.rsl_rl_ppo_cfg:RotaryPendulumBalancePPORunnerCfg",
    },
)

# ============================================================================
# CURRICULUM LEARNING - Stage 1: Swing-up Only
# ============================================================================
gym.register(
    id="Isaac-RotaryPendulum-V2-Balance-Stage1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.balance.__name__}.rotary_pendulum_v2_balance_curriculum:RotaryPendulumV2BalanceStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.balance.agents.__name__}.rsl_rl_ppo_curriculum_cfg:RotaryPendulumStage1PPORunnerCfg",
    },
)

# ============================================================================
# CURRICULUM LEARNING - Stage 2: Balance + Heading Tracking
# ============================================================================
gym.register(
    id="Isaac-RotaryPendulum-V2-Balance-Stage2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.balance.__name__}.rotary_pendulum_v2_balance_curriculum:RotaryPendulumV2BalanceStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.balance.agents.__name__}.rsl_rl_ppo_curriculum_cfg:RotaryPendulumStage2PPORunnerCfg",
    },
)
