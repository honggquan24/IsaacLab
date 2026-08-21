# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Robot bipedal wheel (V5) — hai bánh, chân 5 khâu (5-bar, USD export từ Onshape).

Chạy một lần trước khi dùng, để sinh ``usd/wheeled_biped_fixed.usd``:

    ./isaaclab.sh -p scripts/ute/wheeled_biped/prepare_usd.py


Cách đọc lệnh quay video
------------------------
``--video_length`` đếm theo BƯỚC ĐIỀU KHIỂN, không phải giây. Tần số điều khiển
= 1 / (sim.dt × decimation), ghi kèm ở từng task bên dưới.
Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``.
``--load_run`` lấy checkpoint mới nhất trong thư mục run đó; muốn chỉ đúng một
checkpoint thì thay bằng ``--checkpoint <đường/dẫn/model_xxx.pt>``.
Bỏ ``--headless`` nếu muốn xem cửa sổ Isaac Sim trong lúc ghi.

Isaac-Wheeled-Biped-Wheel — locomotion bám vận tốc, bản MIMIC (policy ra 2 hip)
    100 Hz (sim.dt 1/200, decimation 2) → 60 s = 6000 step, 120 s = 12000 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Wheeled-Biped-Wheel --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Wheeled-Biped-Wheel-Play --num_envs 4 --headless \
        --video --video_length 6000 \
        --checkpoint logs/rsl_rl/legged_v5_wheel_mimic/2026-06-17_03-39-14/model_10799.pt

Isaac-Wheeled-Biped-Wheel-Play — như trên nhưng 4 env + khung nhìn gần, dành để quay
    Dùng chung checkpoint và experiment_name ``legged_v5_wheel_mimic`` với bản train.

Isaac-Wheeled-Biped-Wheel-NoMimic — policy điều khiển cả 4 hip
    100 Hz → 60 s = 6000 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Wheeled-Biped-Wheel-NoMimic --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Wheeled-Biped-Wheel-NoMimic --num_envs 4 --headless \
        --video --video_length 6000 --load_run <tên_run>

Isaac-Wheeled-Biped-Wheel-PIANN — mạng xuất Kp/Ki/Kd, PID quy ra lệnh bánh
    100 Hz → 60 s = 6000 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Wheeled-Biped-Wheel-PIANN --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Wheeled-Biped-Wheel-PIANN --num_envs 4 --headless \
        --video --video_length 6000 --load_run <tên_run>

Isaac-Wheeled-Biped-Navigation — tầng cao ra lệnh vận tốc, tầng thấp là policy đã train
    10 Hz (decimation 20) → 60 s = 600 step, 120 s = 1200 step
    Trước khi train: sửa ``policy_path`` trong ``navigation/navigation_env_cfg.py``
    trỏ tới ``exported/policy.pt`` của run locomotion muốn dùng.

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Wheeled-Biped-Navigation --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Wheeled-Biped-Navigation --num_envs 4 --headless \
        --video --video_length 600 --load_run <tên_run>

Isaac-Wheeled-Biped-Warehouse-Nav — né vật cản trong kho bằng LiDAR
    10 Hz → 60 s = 600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Wheeled-Biped-Warehouse-Nav --num_envs 512 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Wheeled-Biped-Warehouse-Nav --num_envs 2 --headless \
        --video --video_length 600 --load_run <tên_run>

Isaac-Wheeled-Biped-Obstacle-Nav — né vật cản trên terrain sinh thủ tục
    10 Hz → 60 s = 600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Wheeled-Biped-Obstacle-Nav --num_envs 1024 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Wheeled-Biped-Obstacle-Nav --num_envs 4 --headless \
        --video --video_length 600 --load_run <tên_run>

Lái tay bằng bàn phím (pygame) để quay cảnh điều khiển chủ động:

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_teleop_wheeled_biped.py \
        --task Isaac-Wheeled-Biped-Wheel --num_envs 1 \
        --checkpoint logs/rsl_rl/legged_v5_wheel_mimic/2026-06-17_03-39-14/model_10799.pt
"""

import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Wheeled-Biped-Wheel",          # bản MIMIC (policy ra 2 hip, mimic auto)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_env_cfg:WheeledBipedWheelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Wheel-Play",     # như bản MIMIC nhưng ít env + camera bám (quay video)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_env_cfg:WheeledBipedWheelPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Wheel-NoMimic",  # bản KHÔNG mimic (policy ra cả 4 hip)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_nomimic_env_cfg:WheeledBipedWheelNoMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelNoMimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Wheel-PIANN",    # bánh = PI-ANN (mạng xuất Kp/Ki/Kd, PID quy ra lệnh bánh)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion.wheel_piann_env_cfg:WheeledBipedWheelPIANNEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWheelPIANNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Navigation",     # tầng cao: command pos → goal, low-level pretrained
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_env_cfg:WheeledBipedNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Warehouse-Nav",  # tầng cao + LiDAR: né vật cản trong kho, tới đích
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.warehouse_nav_env_cfg:WheeledBipedWarehouseNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedWarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Wheeled-Biped-Obstacle-Nav",   # tầng cao + LiDAR: né vật cản trên terrain sinh thủ tục (train song song)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.obstacle_nav_env_cfg:WheeledBipedObstacleNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledBipedObstacleNavPPORunnerCfg",
    },
)

# Re-export ở cuối file (sau các lần đăng ký) để lỗi import của một env cfg không làm
# hỏng việc đăng ký các task còn lại.
from .locomotion.wheel_env_cfg import WheeledBipedWheelEnvCfg, WheeledBipedWheelPlayEnvCfg  # noqa: E402
from .navigation import WheeledBipedNavigationEnvCfg, WheeledBipedWarehouseNavEnvCfg  # noqa: E402
