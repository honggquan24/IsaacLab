"""Biped RL-tuned cascade PID — Legged V3 URDF (wheeled inverted pendulum).

Training order:
    # Bước 1 — Train vòng trong (tilt + yaw PID)
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Inner-Tilt --num_envs 512 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Biped-Inner-Tilt --num_envs 4
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
