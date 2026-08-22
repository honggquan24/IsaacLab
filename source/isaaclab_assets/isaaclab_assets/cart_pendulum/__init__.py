# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Con lắc đơn trên xe đẩy (CAD Onshape).

Ray dài 1.11 m nằm dọc trục Y, xe trượt trên ray, con lắc dài 0.22 m gắn trên xe và quay tự
do. Ở góc khớp 0 con lắc thõng xuống; tư thế đứng là π và được đặt làm vị trí khớp mặc định,
nên reward đo lệch so với mặc định chứ không so với 0.

Chuẩn bị USD
------------
``usd/cart_pendulum_base.usd`` là bản Onshape thô, ``usd/cart_pendulum_cfg.usd`` là bản đã vá
và là bản env dùng. Sinh lại bản vá bằng::

    ./isaaclab.sh -p scripts/ute/prepare_usd.py --verify

Cách đọc lệnh quay video
------------------------
``--video_length`` đếm theo BƯỚC ĐIỀU KHIỂN, không phải giây. Tần số điều khiển
= 1 / (sim.dt × decimation), ghi kèm ở từng task bên dưới.
Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``.
``--load_run`` lấy checkpoint mới nhất trong thư mục run đó; muốn chỉ đúng một
checkpoint thì thay bằng ``--checkpoint <đường/dẫn/model_xxx.pt>``.
Bỏ ``--headless`` nếu muốn xem cửa sổ Isaac Sim trong lúc ghi.

Isaac-Cart-Pendulum — giữ con lắc đứng, xe bám quanh giữa ray
    60 Hz (sim.dt 1/60, decimation 1) → 60 s = 3600 step, 120 s = 7200 step

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum --num_envs 4096 --video --video_length 3600 --rendering_mode quality --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>

Isaac-Cart-Pendulum-Position — vừa giữ con lắc vừa chạy tới mốc vị trí được lệnh
    60 Hz (sim.dt 1/60, decimation 1) → 60 s = 3600 step, 120 s = 7200 step
    Mốc đổi sau mỗi 3–5 s, hiện bằng quả cầu đỏ trên ray (``debug_vis``).

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cart-Pendulum-Position --num_envs 4096 --headless

    Quay video thì dùng task ``-Play`` (episode 60 s nên clip không bị reset giữa chừng),
    nó dùng chung thư mục log ``cartpole_v1_position_ppo`` với task train ở trên:

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Cart-Pendulum-Position-Play --num_envs 4 --headless \
        --video --video_length 3600 --load_run <tên_run>
"""

import gymnasium as gym

from . import agents
from .cart_pendulum_cfg import *  # noqa: F403
from .cart_pendulum_env_cfg import *  # noqa: F403

gym.register(
    id="Isaac-Cart-Pendulum",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_env_cfg:CartPendulumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Cart-Pendulum-Position",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_env_cfg:CartPendulumPositionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumPositionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Cart-Pendulum-Position-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_pendulum_env_cfg:CartPendulumPositionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartPendulumPositionPPORunnerCfg",
    },
)
