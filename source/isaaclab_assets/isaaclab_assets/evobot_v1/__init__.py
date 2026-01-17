import gymnasium as gym
from . import balance, navigation

# ============================================================================
# EVOBOT V1 - Isaac Lab Training & Evaluation Guide
# ============================================================================
#
# This file registers all Evobot V1 tasks for Isaac Lab.
# For detailed documentation, see: EVOBOT.md
#
# IMPORTANT: Add to source/isaaclab_assets/isaaclab_assets/__init__.py:
#   from .evobot_v1 import *
#
# ============================================================================
# QUICK START - Training Commands
# ============================================================================
#
# 1. BALANCE TASK (Standing upright)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Train (production):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Balance \
#       --num_envs 1024 \
#       --headless --rendering_mode performance
#
#   Train (debug with 3 envs):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Balance \
#       --num_envs 3 \
#       --rendering_mode performance
#
#   Continue training from checkpoint:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Balance \
#       --num_envs 1024 \
#       --resume --load_run=<run_name> \
#       --checkpoint=model_<num>.pt \
#       --video --rendering_mode performance \
#       --headless
#
#   Evaluate (test trained model):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Balance \
#       --num_envs 4 \
#       'agent.load_run=<run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#
# ============================================================================
# 2. VELOCITY BALANCE TASK (Balance + velocity following)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Train (production):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Velocity \
#       --num_envs 7000 \
#       --headless --rendering_mode performance
#
#   Train (debug):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Velocity \
#       --num_envs 3 \
#       --rendering_mode performance
#
#   Continue training from checkpoint:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Velocity \
#       --num_envs 7000 \
#       --resume --load_run=2026-01-17_15-33-57 \
#       --checkpoint=model_40.pt \
#       --video --rendering_mode performance \
#       --headless
#
#   Evaluate:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Velocity \
#       --num_envs 4 \
#       'agent.load_run=<run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#
# ============================================================================
# 3. LOCOMOTION-MANIPULATION TASK (Balance + velocity + arm control)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Train (production):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Locomotion-Manipulation \
#       --num_envs 1024 \
#       --headless --rendering_mode performance
#
#   Train (debug):
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Locomotion-Manipulation \
#       --num_envs 3 \
#       --rendering_mode performance
#
#   Evaluate:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Locomotion-Manipulation \
#       --num_envs 4 \
#       'agent.load_run=<run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#
# ============================================================================
# 4. HIERARCHICAL VELOCITY PRETRAINED (Uses pre-trained velocity policy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Architecture:
#   - Low-level policy (pre-trained): Balance + velocity tracking → joint control
#   - High-level policy (train new): Velocity command generation
#
#   STEP 1: Train velocity balance task first (if not already done)
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Velocity \
#       --num_envs 7000 \
#       --headless --rendering_mode performance
#
#   STEP 2: Export velocity balance policy
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Velocity --num_envs 1 \
#       'agent.load_run=2026-01-15_01-19-39' \
#       'agent.load_checkpoint="model_100.pt"'
#   → Policy saved to: logs/rsl_rl/evobot_v1_velocity/<run_name>/exported/policy.pt
#
#   STEP 3: Update policy path in hierarchical_vel_env_cfg.py
#   Edit: navigation/velocity/hierarchical_vel_env_cfg.py
#   Find: policy_path = "logs/rsl_rl/evobot_v1_velocity/2026-01-14_11-14-00/exported/policy.pt"
#   Update with exported policy path from STEP 2
#
#   STEP 4: Train high-level velocity policy
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Velocity-Pretrained \
#       --num_envs 512 \
#       --headless --rendering_mode performance
#
#   STEP 5: Evaluate
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Velocity-Pretrained-Play \
#       --num_envs 16 \
#       'agent.load_run=<run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#
# ============================================================================
# 5. PID-BASED VELOCITY CONTROL (Classical control instead of RL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Architecture:
#   - High-level RL policy: Outputs velocity commands (vx, wz)
#   - Low-level PID controller: Converts (vx, wz) → wheel torques
#
#   Advantages:
#   - No need to train low-level policy
#   - Simpler action space (2D instead of 5D)
#   - Faster training (deterministic low-level control)
#   - Easy to tune (PID gains in velocity_pid_env_cfg.py)
#
#   Train:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Velocity-PID \
#       --num_envs 1024 \
#       --headless --rendering_mode performance
#
#   Evaluate:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Velocity-PID \
#       --num_envs 4 \
#       'agent.load_run=<run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#
#   Tune PID gains (edit velocity_pid_env_cfg.py):
#   - kp_linear, ki_linear, kd_linear: Linear velocity control
#   - kp_angular, ki_angular, kd_angular: Angular velocity control
#   - wheel_base: Distance between wheels (0.135m)
#   - wheel_radius: Wheel radius (0.0325m)
#
# ============================================================================
# 6. HIERARCHICAL NAVIGATION TASK (Requires pre-trained balance policy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   STEP 1: Train balance task first (if not already done)
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Balance \
#       --num_envs 1024 \
#       --headless --rendering_mode performance
#
#   STEP 2: Export balance policy
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Balance --num_envs 1 \
#       'agent.load_run=<balance_run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#   → Policy saved to: logs/rsl_rl/<algo>/<task>/<run_name>/exported/policy.pt
#
#   STEP 3: Update policy path in hierarchical_env_cfg.py
#   Edit: navigation/hierarchical/hierarchical_env_cfg.py
#   Find: policy_path = "logs/.../exported/policy.pt"
#   Update with exported policy path from STEP 2
#
#   STEP 4: Train navigation
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task=Isaac-Evobot-V1-Navigation-Hierarchical \
#       --num_envs 512 \
#       --headless --rendering_mode performance
#
#   STEP 5: Evaluate navigation
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Isaac-Evobot-V1-Navigation-Hierarchical-Play \
#       --num_envs 16 \
#       'agent.load_run=<nav_run_name>' \
#       'agent.load_checkpoint="model_500.pt"'
#
# ============================================================================
# USEFUL UTILITIES
# ============================================================================
#
#   List all registered environments:
#   ./isaaclab.sh -p scripts/environments/list_envs.py | grep Evobot
#
#   Test environment loading:
#   ./isaaclab.sh -p ./source/isaaclab_assets/isaaclab_assets/evobot_v1/tests/run_robot_rl_env.py
#
#   View TensorBoard training logs:
#   ./isaaclab.sh -p -m tensorboard.main --logdir logs
#
# ============================================================================

# ============================================================================
# ENVIRONMENT REGISTRATIONS
# ============================================================================

# Balance Task: Isaac-Evobot-V1-Balance
gym.register(
    id="Isaac-Evobot-V1-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{balance.__name__}.evobot_v1_balance_env_cfg:EvobotV1BalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{balance.agents.__name__}.rsl_rl_ppo_cfg:EvobotBalancePPORunnerCfg"
    },
)

# Velocity Balance Task: Isaac-Evobot-V1-Velocity
gym.register(
    id="Isaac-Evobot-V1-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.velocity.__name__}.velocity_env_cfg:EvobotV1VelocityBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotVelocityPPORunnerCfg",
    },
)

# Locomotion-Manipulation Task: Isaac-Evobot-V1-Locomotion-Manipulation
gym.register(
    id="Isaac-Evobot-V1-Locomotion-Manipulation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.locomotion_manipulation.__name__}.loc_man_env_cfg:EvobotV1LocomotionManipulationBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.locomotion_manipulation.agents.__name__}.rsl_rl_ppo_cfg:EvobotLocomotionManipulationPPORunnerCfg",
    },
)

# Hierarchical Velocity Pretrained Task: Isaac-Evobot-V1-Velocity-Pretrained
# Uses pre-trained low-level balance policy, trains high-level velocity command policy
gym.register(
    id="Isaac-Evobot-V1-Velocity-Pretrained",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.velocity.__name__}.hierarchical_vel_env_cfg:EvobotV1VelocityPretrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotVelocityPretrainPPORunnerCfg",
    },
)

# Hierarchical Velocity Pretrained Evaluation: Isaac-Evobot-V1-Velocity-Pretrained-Play
gym.register(
    id="Isaac-Evobot-V1-Velocity-Pretrained-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.velocity.__name__}.hierarchical_vel_env_cfg:EvobotV1VelocityPretrainedEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{navigation.velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotVelocityPretrainPPORunnerCfg",
    },
)

# Hierarchical Navigation Task: Isaac-Evobot-V1-Navigation-Hierarchical
gym.register(
    id="Isaac-Evobot-V1-Navigation-Hierarchical",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.hierarchical.__name__}.hierarchical_env_cfg:EvobotV1NavigationPretrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.hierarchical.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)

# Hierarchical Navigation Evaluation: Isaac-Evobot-V1-Navigation-Hierarchical-Play
gym.register(
    id="Isaac-Evobot-V1-Navigation-Hierarchical-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.hierarchical.__name__}.hierarchical_env_cfg:EvobotV1NavigationPretrainedEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{navigation.hierarchical.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)

# PID-based Velocity Control: Isaac-Evobot-V1-Velocity-PID
# Uses PID controller to convert velocity commands to wheel torques
gym.register(
    id="Isaac-Evobot-V1-Velocity-PID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.velocity.__name__}.velocity_pid_env_cfg:EvobotV1VelocityPIDEnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotVelocityPIDPPORunnerCfg",
    },
)
