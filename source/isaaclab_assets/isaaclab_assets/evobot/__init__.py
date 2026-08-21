# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""evoBOT — robot hai bánh tự cân bằng có hai tay máy.

Task:
    Isaac-Evobot-Balance            giữ thăng bằng tại chỗ
    Isaac-Evobot-Velocity           bám lệnh vận tốc (kèm cả tay máy)
    Isaac-Evobot-Velocity-Play      như trên, ít env, để xem lại policy
    Isaac-Evobot-Arm-FineTune       tinh chỉnh riêng khớp tay
    Isaac-Evobot-Gripper-FineTune   tinh chỉnh riêng kẹp
    Isaac-Evobot-Navigation         tầng cao tới đích, dùng policy vận tốc đã train
    Isaac-Evobot-Navigation-Play    như trên, ít env
    Isaac-Evobot-Manipulation       vừa di chuyển vừa thao tác

Lệnh train/play chi tiết cho từng task: xem ``docs/ute/evobot_tasks.md``.
"""

import gymnasium as gym

from . import balance, manipulation, navigation, velocity  # noqa: F401

gym.register(
    id="Isaac-Evobot-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{balance.__name__}.balance_env_cfg:EvobotBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{balance.agents.__name__}.rsl_rl_ppo_cfg:EvobotBalancePPORunnerCfg",
    },
)

# Velocity Balance Task: Isaac-Evobot-Velocity
gym.register(
    id="Isaac-Evobot-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{velocity.__name__}.velocity_env_cfg:EvobotVelocityBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotVelocityPPORunnerCfg",
    },
)

# Velocity Balance Task (Play - no noise/disturbances): Isaac-Evobot-Velocity-Play
gym.register(
    id="Isaac-Evobot-Velocity-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{velocity.__name__}.velocity_env_cfg_play:EvobotVelocityBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotVelocityPPORunnerCfg",
    },
)

# Gripper Fine-tuning Task: Isaac-Evobot-Gripper-FineTune
gym.register(
    id="Isaac-Evobot-Gripper-FineTune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{velocity.__name__}.velocity_env_cfg_gripper_finetune:EvobotGripperFineTuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotGripperFineTunePPORunnerCfg",
    },
)

# Arm Fine-tuning Task: Isaac-Evobot-Arm-FineTune
gym.register(
    id="Isaac-Evobot-Arm-FineTune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{velocity.__name__}.velocity_env_cfg_arm_finetune:EvobotArmFineTuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{velocity.agents.__name__}.rsl_rl_ppo_cfg:EvobotArmFineTunePPORunnerCfg",
    },
)

# Navigation Task: Isaac-Evobot-Navigation
gym.register(
    id="Isaac-Evobot-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.__name__}.navigation_env_cfg:EvobotNavigationPretrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{navigation.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)

# Navigation Evaluation: Isaac-Evobot-Navigation-Play
gym.register(
    id="Isaac-Evobot-Navigation-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{navigation.__name__}.navigation_env_cfg:EvobotNavigationPretrainedEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{navigation.agents.__name__}.rsl_rl_ppo_cfg:EvobotNavigationPPORunnerCfg",
    },
)

# Locomotion-Manipulation Task: Isaac-Evobot-Locomotion-Manipulation
gym.register(
    id="Isaac-Evobot-Manipulation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{manipulation.__name__}.manipulation_env_cfg:EvobotLocomotionManipulationBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{manipulation.agents.__name__}.rsl_rl_ppo_cfg:EvobotLocomotionManipulationPPORunnerCfg",
    },
)
