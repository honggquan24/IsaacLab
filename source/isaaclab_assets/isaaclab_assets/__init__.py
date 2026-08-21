# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import os

import toml

# Conveniences to other module directories via relative paths
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]

# isort: off
# Thứ tự dưới đây là bắt buộc: cfg robot/sensor của Isaac Lab phải có trước, vì env cfg của
# các dự án tham chiếu tới chúng ngay lúc import.
from .robots import *  # noqa: F403
from .sensors import *  # noqa: F403

# Import package của từng dự án robot để đăng ký task Gymnasium của nó.
from . import balance_car  # noqa: F401
from . import cart_pendulum  # noqa: F401
from . import cart_pendulum_double  # noqa: F401
from . import evobot  # noqa: F401
from . import rotary_pendulum  # noqa: F401
from . import uav  # noqa: F401
from . import wheeled_biped  # noqa: F401

# isort: on
