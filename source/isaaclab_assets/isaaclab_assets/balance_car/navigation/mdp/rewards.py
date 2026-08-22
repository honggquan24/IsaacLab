# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward bám quỹ đạo cho tầng navigation.

Chỉ còn ba hàm, đúng ba hàm đang được dùng. Bản trước có 15 hàm trong đó 12 hàm là di sản của
thiết kế "chạy tới một đích đứng yên" đã bỏ (``goal_progress_reward``, ``position_reached_bonus``,
``velocity_towards_goal``, ``upright_reward``, ``tilt_penalty``, ...). Chúng không được tham
chiếu ở đâu nhưng vẫn mang docstring nói về một bài toán khác — đọc file là hiểu sai ngay
env đang tối ưu cái gì. Có cái còn giữ trạng thái trong ``env.extras`` dùng chung cho mọi env
và không được dọn lúc reset.

Ba hàm này đọc ``path_command`` do :class:`~.commands.PathCommand` sinh ra, vector
``(num_envs, 4)`` = ``[lệch dọc, lệch ngang, lệch hướng, tốc độ mục tiêu]`` trong hệ heading
của xe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def path_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "path_command",
    std: float = 0.5,
) -> torch.Tensor:
    """Thưởng theo khoảng cách tới điểm mục tiêu đang chạy. Bằng 1 khi trùng khít.

    Dùng ``exp(-d²/std²)`` chứ không phải ``1 - tanh(d/std)`` như mẫu navigation của Isaac Lab:
    ở đó đích nằm cách vài mét nên cần một hàm còn dốc ở XA, còn ở đây mục tiêu luôn ở gần (nó
    xuất phát ngay tại chỗ xe), nên thứ cần là độ phân giải cao QUANH 0 để phân biệt bám sát
    với bám lỏng.

    ``std = 0.5`` nghĩa là lệch 0.5 m còn được 37% điểm, lệch 1 m còn 2%.
    """
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return torch.exp(-torch.square(distance) / std**2)


def path_heading_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "path_command",
    std: float = 0.6,
) -> torch.Tensor:
    """Thưởng khi mũi xe quay đúng chiều tiếp tuyến quỹ đạo.

    Không có term này thì xe vẫn bám được điểm mục tiêu bằng cách **đi lùi** hoặc trượt ngang
    qua các khúc cua — bám đúng vị trí mà nhìn thì sai hoàn toàn.
    """
    command = env.command_manager.get_command(command_name)
    return torch.exp(-torch.square(command[:, 2]) / std**2)


def path_lateral_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Phạt riêng phần lệch NGANG so với quỹ đạo. Dùng với trọng số âm.

    ``path_position_exp`` gộp chung lệch dọc và lệch ngang, nhưng hai cái không tương đương:
    tụt lại phía sau vài chục phân là chuyện bình thường và tự sửa được, còn cắt cua ra ngoài
    đường thì đúng nghĩa là đi sai quỹ đạo.
    """
    command = env.command_manager.get_command(command_name)
    return torch.square(command[:, 1])


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "pose_command",
) -> torch.Tensor:
    """Thưởng theo khoảng cách tới một đích ĐỨNG YÊN, nhân tanh.

    Bản sao của term cùng tên trong ``isaaclab_tasks.manager_based.navigation.mdp`` (không
    import thẳng được vì sẽ tạo vòng import giữa ``isaaclab_assets`` và ``isaaclab_tasks``).

    Chỉ dùng cho ``Isaac-Balance-Car-Navigation`` — nhiệm vụ chạy tới một điểm. Bài bám quỹ đạo
    dùng :func:`path_position_exp`, vì ở đó mục tiêu luôn ở gần và cần độ phân giải quanh 0
    chứ không cần độ dốc ở xa.

    ``command[:, :3]`` là vector tới đích trong hệ thân; lấy ``norm`` nên không phụ thuộc quy
    ước trục của xe.
    """
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :3], dim=1)
    return 1 - torch.tanh(distance / std)
