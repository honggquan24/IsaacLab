# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_angle_r(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 2.0,
):
    """Thưởng khi thân xe dựng thẳng.

    .. important::
        ``target`` là **0.0** chứ không phải 90° như bản CAD cũ. Bản vẽ cũ để xe nằm nghiêng
        nên tư thế đứng ứng với ``|roll| = 90°``; bản mới vẽ thẳng, tư thế đứng là quaternion
        đơn vị nên ``roll = 0``. Để nguyên 90° thì xe bị thưởng vì NẰM NGHIÊNG.
    """
    imu = env.scene["imu"]
    quat = imu.data.quat_w
    roll, _, _ = euler_xyz_from_quat(quat)

    err = torch.abs(roll) - target
    reward = scale * (-0.9 + torch.cos(err))
    return reward


def reward_angle_y(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 2.0,
):
    """Thưởng khi góc yaw bám một hướng TUYỆT ĐỐI trong world.

    Chỉ hợp với bài "đứng yên giữ thăng bằng". Đừng dùng cho bài có lệnh rẽ hay có đích: giữ
    yaw cố định là chống lại đúng việc phải làm. Cả hai task hiện tại đều đã bỏ term này.
    """
    imu = env.scene["imu"]
    quat = imu.data.quat_w
    _, _, yaw = euler_xyz_from_quat(quat)

    err = yaw - target
    reward = scale * (-0.9 + torch.cos(err))
    return reward


def reward_vel(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 0.005,
):
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel

    err1 = torch.abs(joint_vel[:, 0]) - target
    err2 = torch.abs(joint_vel[:, 1]) - target

    reward = torch.exp(-scale * (err1**2 + err2**2))

    return reward


def bonus_reward(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    bonus: float = 0.5,
):
    imu = env.scene["imu"]
    quat = imu.data.quat_w
    roll_, _, _ = euler_xyz_from_quat(quat)
    roll = torch.abs(roll_)

    # dải ±2° quanh tư thế đứng (roll = 0). Bản cũ dò dải 88°–92° vì CAD cũ để xe nằm nghiêng.
    cond = roll < (2 * math.pi / 180)

    reward = torch.where(cond, torch.full_like(roll, bonus), 0.0)
    return reward


def penalty_when_center_of_env_l2(
    env: ManagerBasedRLEnv, target_x: float = 0.0, target_y: float = 0.0, scale: float = 0.1
):
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w - env.scene.env_origins

    err_x = pos[:, 0] - target_x
    err_y = pos[:, 1] - target_y

    penalty = -scale * (err_x**2 + err_y**2)
    return penalty


def reward_li_vel(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 1.0,
):
    imu = env.scene["imu"]
    l_vel = imu.data.lin_vel_b
    y_vel = l_vel[:, 1]
    err = y_vel - target

    reward = torch.exp(-scale * err**2)

    return reward


def reward_roll_rate(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    scale: float = 2.0,
):
    imu = env.scene["imu"]
    roll_rate = imu.data.ang_vel_b[:, 0]

    err = roll_rate - target

    reward = torch.exp(-scale * err**2)
    return reward


"""
Bám lệnh vận tốc — reward của TẦNG THẤP sau khi đổi từ "đứng yên giữ thăng bằng"
sang "vừa giữ thăng bằng vừa chạy theo lệnh".
"""


def track_lin_vel_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.25,
    forward_sign: float = 1.0,
):
    """Thưởng khi vận tốc tiến của xe bám lệnh ``vx``.

    Lấy ``imu.data.lin_vel_b[:, 1]`` làm vận tốc tiến — đúng thành phần mà quan sát
    :func:`~..observations.lin_vel_b` đang đưa vào mạng, nên reward và quan sát nói cùng một
    thứ. Thân xe bị xoay 90° quanh trục roll (tư thế đứng là ``|roll| = 90°``) nên KHÔNG dùng
    được ``root_lin_vel_b[:, 0]`` như các task locomotion của Isaac Lab.

    .. important::
        ``forward_sign`` phải kiểm bằng mắt một lần. Nếu +y của IMU chỉ về phía sau xe thì
        lệnh tiến sẽ làm xe lùi, mà reward vẫn báo bám tốt — sai kiểu này không lộ ra trong
        log, chỉ thấy khi nhìn robot chạy. Thấy ngược thì đặt -1.0.
    """
    imu = env.scene["imu"]
    command = env.command_manager.get_command(command_name)
    lin_vel = forward_sign * imu.data.lin_vel_b[:, 1]
    return torch.exp(-torch.square(command[:, 0] - lin_vel) / std**2)


def track_ang_vel_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    turn_sign: float = 1.0,
):
    """Thưởng khi tốc độ quay của xe bám lệnh ``wz``.

    Đo trong hệ WORLD (``root_ang_vel_w[:, 2]``) chứ không phải hệ thân: thân xe xoay 90° nên
    trục nào của nó là trục quay đứng còn tuỳ tư thế, trong khi z của world thì luôn là trục
    quay của việc rẽ.
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    ang_vel = turn_sign * robot.data.root_ang_vel_w[:, 2]
    return torch.exp(-torch.square(command[:, 2] - ang_vel) / std**2)


"""
Giữ nắp xe nằm ngang tuyệt đối.

Đo bằng **vector trọng lực trong hệ thân** (``projected_gravity_b``) chứ không phải góc Euler.
Lý do bỏ cách cũ (``reward_angle_r`` dùng ``euler_xyz_from_quat``):

* Euler chỉ bắt được ROLL. Nắp xe nghiêng theo pitch — chúi mũi/ngóc đuôi — thì roll vẫn 0 và
  reward vẫn báo phẳng. ``projected_gravity_b`` bắt cả hai trục cùng lúc.
* ``cos(err)`` có đạo hàm bậc nhất bằng 0 quanh 0 nên gần tư thế phẳng nó gần như PHẲNG LÌ:
  chênh 0° hay 3° cho điểm gần bằng nhau, policy không có lý do gì phải chính xác nốt vài độ
  cuối. Đó đúng là biểu hiện "nắp xe lắc lư mãi không đứng yên".
* Euler còn có điểm gãy và gimbal; vector trọng lực thì không.

Khi thân dựng thẳng, trọng lực trong hệ thân là ``(0, 0, -1)``. Hai thành phần ngang ``x``,
``y`` chính là ``sin`` của độ nghiêng theo từng trục, nên chúng bằng 0 **khi và chỉ khi** trục
z của thân trùng trục z của world — tức nắp xe nằm ngang tuyệt đối.
"""


def cover_flat_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phạt bình phương độ nghiêng của nắp xe. Bằng ``sin²(θ)``, chỉ bằng 0 khi phẳng tuyệt đối.

    Dùng làm term PHẠT (trọng số âm). Đây là tín hiệu rộng: còn dốc ở mọi độ nghiêng nên luôn
    kéo về phẳng, kể cả lúc xe đang nghiêng nhiều.

    ``cover`` và ``base`` nằm chung một thân cứng ``Group_1``, nên nắp phẳng chính là thân
    dựng — không cần đo riêng prim nắp.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def cover_flat_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Thưởng nhọn cho vài độ cuối. Dùng làm term THƯỞNG (trọng số dương).

    ``cover_flat_l2`` một mình không đủ để có "tuyệt đối phẳng": đạo hàm của nó tỉ lệ với độ
    nghiêng nên càng gần phẳng nó càng hết lực kéo. Term này bù đúng chỗ đó — với
    ``std = 0.05`` thì đỉnh chỉ rộng khoảng ±3°, ra ngoài là rơi rất nhanh, nên phần thưởng
    chỉ thật sự ăn được khi nắp gần như song song mặt sàn.

    Đây cũng là bản trơn của ``bonus_reward`` cũ. Bản cũ dùng điều kiện cứng "nằm trong dải
    ±2°" — một bậc thang, mà bậc thang thì không có đạo hàm để policy bám theo, nó chỉ biết
    mình rơi ra ngoài chứ không biết cách nào để vào.
    """
    return torch.exp(-cover_flat_l2(env, asset_cfg) / std**2)
