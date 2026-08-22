# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_when_fall(env: ManagerBasedRLEnv):
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w

    roll, _, _ = euler_xyz_from_quat(quat)

    # Tư thế đứng của bản CAD mới là quaternion đơn vị → roll = 0. Bản cũ vẽ xe nằm nghiêng
    # nên chỗ này từng là pi/2; giữ nguyên số cũ thì mọi env bị kết thúc ngay tại bước reset.
    upright = 0.0
    # 40° chứ không phải 50°: với ma sát 1.0 thì gia tốc lớn nhất là g, nên góc còn cứu được
    # tối đa thoả tan(θ) = μ, tức 45° — và ở sát 45° thì cần vô hạn thời gian. Đặt ngưỡng
    # trên mức khả thi chỉ khiến policy phải học từ những thế cờ vật lý không cho phép cứu.
    threshold = math.pi / 180 * 40

    # euler_xyz_from_quat trả về (-pi, pi] nên quanh 0 không có điểm gãy, so trực tiếp được
    terminate = torch.abs(roll - upright) > threshold
    return terminate
