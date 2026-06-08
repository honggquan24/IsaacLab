"""Biped RL-tuned cascade PID — Legged V3 URDF (wheeled inverted pendulum).

Training order:
    # Bước 1 — Train vòng trong (tilt + yaw PID)
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Inner-Tilt --num_envs 512 --headless

    # Bước 2a — Train vòng ngoài (velocity PID → tilt setpoint)
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Outer-Vel-PID --num_envs 512 --headless

    # Bước 2b — Train vòng ngoài (direct tilt setpoint, không qua PID)
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Outer-Vel-Direct --num_envs 512 --headless
"""

from .biped_cfg import *
from . import mdp

import gymnasium as gym
from . import agents

gym.register(
    id="Biped-Inner-Tilt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_control.inner_tilt_env_cfg:BipedInnerTiltEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedInnerTiltRunnerCfg",
    },
)

gym.register(
    id="Biped-Outer-Vel-PID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_control.outer_vel_pid_env_cfg:BipedOuterVelPIDEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedOuterVelPIDRunnerCfg",
    },
)

gym.register(
    id="Biped-Unified-Vel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_control.unified_vel_pid_env_cfg:BipedUnifiedVelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedUnifiedVelRunnerCfg",
    },
)

gym.register(
    id="Biped-Outer-Vel-Direct",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_control.outer_vel_direct_env_cfg:BipedOuterVelDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedOuterVelDirectRunnerCfg",
    },
)
