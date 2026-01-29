"""Legged Robot V3 - 2-legged wheeled robot velocity tracking.

Robot: robot_legged_v2 from Onshape (exported as robot_legged_v3.usd)
Structure:
  - base
  - left_leg: left_hip_joint, left_thigh_joint, left_knee_joint, left_wheel_joint
  - right_leg: right_hip_joint, right_thigh_joint, right_knee_joint, right_wheel_joint

Tasks:
  - Isaac-Legged-V3-Velocity: Velocity tracking with balance

Train (headless):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Velocity \\
        --num_envs 4096 --headless

Train (visual debug):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Velocity \\
        --num_envs 64

Resume training:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Velocity \\
        --num_envs 4096 \\
        --resume --load_run=<run_name> --checkpoint=model_100.pt

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V3-Velocity \\
        --num_envs 4 \\
        'agent.load_run=<run_name>' 'agent.load_checkpoint="model_100.pt"'
"""

from .legged_v3_cfg import *
from .legged_v3_velocity_env_cfg import *

import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Legged-V3-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.legged_v3_velocity_env_cfg:LeggedV3VelocityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeggedV3VelocityPPORunnerCfg",
    },
)
