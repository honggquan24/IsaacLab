# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Xe hai bánh tự cân bằng — bám lệnh vận tốc và bám quỹ đạo.

Ba task, dựng theo hai mẫu chuẩn của Isaac Lab
==============================================

======================================  ==============================================  ======
task                                    mẫu Isaac Lab                                   Hz
======================================  ==============================================  ======
``Isaac-Balance-Car``                   ``manager_based/locomotion/velocity``           50
``Isaac-Balance-Car-Navigation``        cùng mẫu, đổi lệnh vận tốc → lệnh vị trí        50
``Isaac-Balance-Car-Navigation-Pretrained``  ``manager_based/navigation`` (cascade)     10
======================================  ==============================================  ======

Toàn bộ MDP của tầng thấp dùng term có sẵn của ``isaaclab.envs.mdp`` — không còn observation,
reward, termination hay command tự viết. Phần tự viết còn lại đúng hai thứ, và cả hai đều là
thứ Isaac Lab không có: :class:`~.navigation.mdp.commands.PathCommand` (mục tiêu chạy trên
đường cong kín) và ba hàm reward bám quỹ đạo đi kèm.

Hướng tiến của xe là **+Y của thân**, không phải +X như quy ước locomotion. Chỗ duy nhất phải
biết điều đó là dải lệnh trong ``CommandsCfg``: thành phần tiến nằm ở ``lin_vel_y``. Xem
:mod:`.balance_env_cfg`.

Thứ tự chạy
===========
Task cascade cần policy tầng thấp đã export. ``train.py`` **không** sinh ra file đó, chỉ
``play.py`` mới sinh — nên bắt buộc phải chạy play của ``Isaac-Balance-Car`` ít nhất một lần
trước khi train cascade. Đường dẫn tự dò theo run mới nhất, không phải sửa config bằng tay.

Cách đọc lệnh quay video
========================
``--video_length`` đếm theo **bước điều khiển**, không phải giây; tần số ghi ở bảng trên.
Video xuất ra ``logs/rsl_rl/<experiment_name>/<run>/videos/play/``. ``--load_run`` lấy
checkpoint mới nhất trong run đó; muốn chỉ đúng một checkpoint thì dùng
``--checkpoint <đường/dẫn/model_xxx.pt>``. Bỏ ``--headless`` để xem cửa sổ Isaac Sim.

1. Isaac-Balance-Car — giữ thăng bằng, bám lệnh vận tốc (50 Hz → 60 s = 3000 step)
---------------------------------------------------------------------------------
::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Balance-Car --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Balance-Car-Play --num_envs 16 \
        --video --video_length 3000 --load_run <tên_run>

Mũi tên XANH LÁ là lệnh, XANH DƯƠNG là vận tốc thật, cùng hệ nên nhìn là biết bám tốt hay
không. Cả hai xoay theo thân xe — đúng, vì chúng là đại lượng trong hệ thân.

2. Isaac-Balance-Car-Navigation — chạy tới đích, học từ đầu (50 Hz → 60 s = 3000 step)
--------------------------------------------------------------------------------------
Một mạng vừa cân bằng vừa điều hướng. Khó hơn hẳn bản cascade, để đối chứng.
::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Balance-Car-Navigation --num_envs 4096 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Balance-Car-Navigation-Play --num_envs 16 \
        --video --video_length 3000 --load_run <tên_run>

3. Isaac-Balance-Car-Navigation-Pretrained — BÁM QUỸ ĐẠO (10 Hz → 60 s = 600 step)
----------------------------------------------------------------------------------
Mục tiêu là một điểm **chạy liên tục** trên đường tròn hoặc hình số 8 (bán kính 1-2 m,
0.4-0.9 m/s), không phải đích đứng yên. Quả cầu đỏ = mục tiêu đang chạy, chuỗi chấm xanh =
nguyên hình quỹ đạo. Điều kiện: đã train ``Isaac-Balance-Car`` **và** chạy ``play.py`` của nó
ít nhất một lần.
::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Balance-Car-Navigation-Pretrained --num_envs 2048 --headless

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Balance-Car-Navigation-Pretrained-Play --num_envs 16 \
        --video --video_length 600 --load_run <tên_run>

Chuẩn bị USD
============
``usd/balance_car_base.usd`` là bản Onshape thô, ``usd/balance_car_cfg.usd`` là bản đã vá::

    ./isaaclab.sh -p scripts/ute/prepare_usd.py --package balance_car \
        --floating-base --base-body Group_1 --max-angular-velocity 40 --verify
"""

import gymnasium as gym

from . import agents
from .balance_car_cfg import *  # noqa: F401, F403
from .balance_env_cfg import *  # noqa: F401, F403
from .navigation import agents as nav_agents

##
# 1. Giữ thăng bằng + bám lệnh vận tốc
##

gym.register(
    id="Isaac-Balance-Car",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:BalanceCarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceCarPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Balance-Car-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:BalanceCarEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceCarPPORunnerCfg",
    },
)

##
# 2. Chạy tới đích, học từ đầu
##

gym.register(
    id="Isaac-Balance-Car-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_env_cfg:BalanceCarNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Balance-Car-Navigation-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation.navigation_env_cfg:BalanceCarNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPPORunnerCfg",
    },
)

##
# 3. Bám quỹ đạo trên policy thăng bằng đã train
##

gym.register(
    id="Isaac-Balance-Car-Navigation-Pretrained",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.navigation.navigation_pretrained_env_cfg:BalanceCarNavigationPretrainedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPretrainedPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Balance-Car-Navigation-Pretrained-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.navigation.navigation_pretrained_env_cfg:BalanceCarNavigationPretrainedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": f"{nav_agents.__name__}.rsl_rl_ppo_cfg:BalanceCarNavigationPretrainedPPORunnerCfg",
    },
)
