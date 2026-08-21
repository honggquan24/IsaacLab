# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""evoBOT — robot hai bánh tự cân bằng có hai tay máy.

Lệnh train/play chi tiết cho từng biến thể cũ: xem ``docs/ute/evobot_tasks.md``.


Cách đọc lệnh quay video
------------------------
``--video_length`` đếm theo BƯỚC ĐIỀU KHIỂN, không phải giây. Tần số điều khiển
= 1 / (sim.dt × decimation), ghi kèm ở từng task bên dưới.
Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``.
``--load_run`` lấy checkpoint mới nhất trong thư mục run đó; muốn chỉ đúng một
checkpoint thì thay bằng ``--checkpoint <đường/dẫn/model_xxx.pt>``.
Bỏ ``--headless`` nếu muốn xem cửa sổ Isaac Sim trong lúc ghi.

Isaac-Evobot-Balance — giữ thăng bằng tại chỗ
    60 Hz (sim.dt 1/60, decimation 1) → 60 s = 3600 step; mỗi episode 10 s = 600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Evobot-Balance --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Evobot-Balance --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Evobot-Velocity — bám lệnh vận tốc, có cả tay máy
    60 Hz → 60 s = 3600 step; mỗi episode 10 s = 600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Evobot-Velocity --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Evobot-Velocity-Play --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Evobot-Velocity-Play — như trên, 60 s mỗi episode nên hợp để quay liền mạch
    60 Hz → mỗi episode 60 s = 3600 step (quay 3600 step là trọn một episode)

Isaac-Evobot-Arm-FineTune — tinh chỉnh riêng khớp tay
    60 Hz → 60 s = 3600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Evobot-Arm-FineTune --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Evobot-Arm-FineTune --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Evobot-Gripper-FineTune — tinh chỉnh riêng kẹp
    60 Hz → 60 s = 3600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Evobot-Gripper-FineTune --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Evobot-Gripper-FineTune --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Evobot-Manipulation — vừa di chuyển vừa thao tác
    60 Hz → 60 s = 3600 step; mỗi episode 10 s = 600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Evobot-Manipulation --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Evobot-Manipulation --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Evobot-Navigation — tầng cao tới đích, dùng policy vận tốc đã train
    15 Hz (decimation 1×4) → 60 s = 900 step; mỗi episode 5 s = 75 step
    Phải train ``Isaac-Evobot-Velocity`` trước, rồi sửa ``policy_path`` trong
    ``navigation/navigation_env_cfg.py`` trỏ tới ``exported/policy.pt``.

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Evobot-Navigation --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Evobot-Navigation-Play --num_envs 4 --headless \
        --video --video_length 900 --load_run <tên_run>
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
