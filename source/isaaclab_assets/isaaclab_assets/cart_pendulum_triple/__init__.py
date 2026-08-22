# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Con lắc ba trên xe đẩy (CAD Onshape).

Cùng ray và cùng xe với con lắc đơn/kép, ba khâu nối tiếp nhau. Mọi khớp quay bằng 0 là cả
chuỗi thõng thẳng xuống; dựng đứng là ``Revolute_1`` = π, ``Revolute_2`` = ``Revolute_3`` = 0.

Chuỗi trong USD: ``rack -> cart -> pendulum -> pendulum_01 -> pendulum_02``. Ray ở z≈0.65 m,
mỗi khâu 0.22 m nên khi dựng đứng đỉnh chuỗi tới z≈1.31 m.

Chuẩn bị USD
------------
``usd/cart_pendulum_triple_base.usd`` là bản Onshape thô,
``usd/cart_pendulum_triple_cfg.usd`` là bản đã vá và là bản env dùng. Sinh lại bằng::

    ./isaaclab.sh -p scripts/ute/cart_pendulum/prepare_usd.py \
        --package cart_pendulum_triple --verify

Cách đọc lệnh quay video
------------------------
``--video_length`` đếm theo BƯỚC ĐIỀU KHIỂN, không phải giây. Tần số điều khiển
= 1 / (sim.dt × decimation). Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``.

Isaac-Cart-Pendulum-Triple — swing-up rồi giữ ba khâu thẳng đứng
    60 Hz (sim.dt 1/60, decimation 1) → 60 s = 3600 step, 120 s = 7200 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum-Triple --num_envs 4096 --video --video_length 3600 --rendering_mode quality --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum-Triple --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Cart-Pendulum-Triple-Position — bám mốc vị trí, bắt đầu sẵn ở tư thế đứng
    60 Hz → 60 s = 3600 step. Mốc đổi sau mỗi 3–5 s, hiện bằng quả cầu đỏ trên ray.

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum-Triple-Position --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum-Triple-Position-Play --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>
"""

import gymnasium as gym

from . import agents
from .cart_pendulum_triple_cfg import *  # noqa: F403
from .cart_pendulum_triple_env_cfg import *  # noqa: F403

gym.register(
    id="Isaac-Cart-Pendulum-Triple",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_triple_env_cfg:CartPendulumTripleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumTriplePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Cart-Pendulum-Triple-Position",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_triple_env_cfg:CartPendulumTriplePositionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumTriplePositionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Cart-Pendulum-Triple-Position-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_triple_env_cfg:CartPendulumTriplePositionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumTriplePositionPPORunnerCfg",
    },
)
