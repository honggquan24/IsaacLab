# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hai task điều hướng cho xe cân bằng.

* :mod:`.navigation_env_cfg` — học từ đầu, một mạng vừa cân bằng vừa chạy tới đích;
* :mod:`.navigation_pretrained_env_cfg` — cascade, tầng cao bám quỹ đạo trên policy thăng
  bằng đã train.

Task được đăng ký ở ``balance_car/__init__.py``, không phải ở đây.
"""

from . import agents  # noqa: F401
from .navigation_env_cfg import *  # noqa: F401, F403
from .navigation_pretrained_env_cfg import *  # noqa: F401, F403
