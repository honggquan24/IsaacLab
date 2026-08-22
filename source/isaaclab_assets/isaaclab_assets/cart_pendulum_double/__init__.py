# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Con lắc kép trên xe đẩy (CAD Onshape).

Cùng ray 1.11 m dọc trục Y và cùng xe với con lắc đơn, nhưng hai khâu 0.22 m nối tiếp nhau:
``rack -> cart -> pendulum -> pendulum_01``. Mọi khớp quay bằng 0 là cả chuỗi thõng thẳng
xuống; dựng đứng là ``Revolute_1`` = π và ``Revolute_2`` = 0 (khâu 2 thẳng hàng khâu 1).

Bài mặc định là **swing-up**: robot bắt đầu ở tư thế thõng và phải lắc lên rồi giữ. Con lắc
kép swing-up là bài khó — đừng trông đợi nó lên ngay trong vài trăm vòng lặp.

Chuẩn bị USD
------------
``usd/cart_pendulum_double_base.usd`` là bản Onshape thô,
``usd/cart_pendulum_double_cfg.usd`` là bản đã vá và là bản env dùng. Sinh lại bằng::

    ./isaaclab.sh -p scripts/ute/cart_pendulum/prepare_usd.py \
        --package cart_pendulum_double --verify

Cách đọc lệnh quay video
------------------------
``--video_length`` đếm theo BƯỚC ĐIỀU KHIỂN, không phải giây. Tần số điều khiển
= 1 / (sim.dt × decimation), ghi kèm ở từng task bên dưới.
Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``.

Isaac-Cart-Pendulum-Double — swing-up rồi giữ hai khâu thẳng đứng
    30 Hz (sim.dt 1/60, decimation 2) → 60 s = 1800 step, 120 s = 3600 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum-Double --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum-Double --num_envs 4 --headless \
        --video --video_length 1800 --load_run <tên_run>

Isaac-Cart-Pendulum-Double-Position — bám mốc vị trí, bắt đầu sẵn ở tư thế đứng
    30 Hz → 60 s = 1800 step. Mốc đổi sau mỗi 3–5 s, hiện bằng quả cầu đỏ trên ray.

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum-Double-Position --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum-Double-Position-Play --num_envs 4 --headless \
        --video --video_length 1800 --load_run <tên_run>
"""

import gymnasium as gym

from . import agents
from .cart_pendulum_double_cfg import *  # noqa: F403
from .cart_pendulum_double_env_cfg import *  # noqa: F403

gym.register(
    id="Isaac-Cart-Pendulum-Double",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_double_env_cfg:CartPendulumDoubleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumDoublePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Cart-Pendulum-Double-Position",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_double_env_cfg:CartPendulumDoublePositionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumDoublePositionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Cart-Pendulum-Double-Position-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_double_env_cfg:CartPendulumDoublePositionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumDoublePositionPPORunnerCfg",
    },
)
