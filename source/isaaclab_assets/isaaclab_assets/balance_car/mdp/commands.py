# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lệnh vận tốc cho xe cân bằng, với metric đo đúng trục.

Vì sao phải kế thừa lại
-----------------------
``UniformVelocityCommand`` của Isaac Lab tính metric như sau::

    error_vel_xy = norm(vel_command_b[:, :2] - robot.data.root_lin_vel_b[:, :2])

tức là so ``command[0]`` với **body X** và ``command[1]`` với **body Y**. Quy ước đó đúng cho
robot locomotion của Isaac Lab, nơi hướng tiến là +X của body.

Xe này thì hướng tiến là **+Y của body** (xem ``BALANCE_CAR_CFG``). Nên metric gốc so lệnh
tiến với vận tốc NGANG, và so lệnh ngang (luôn bằng 0) với vận tốc TIẾN. Hậu quả: kể cả một
policy bám lệnh hoàn hảo cũng cho ``error_vel_xy = |cmd| · √2`` chứ không bao giờ về 0 — con
số đó **không đọc được**, không phân biệt được policy tốt với policy đứng im.

Đây đúng loại bẫy đã tốn nhiều thời gian ở robot V5: một metric trông có vẻ hợp lý, thay đổi
theo quá trình học, nhưng đo sai đại lượng nên dẫn người đọc đi sai hướng. Sửa metric rẻ hơn
nhiều so với việc train mấy tiếng rồi mới phát hiện.
"""

from __future__ import annotations

import torch

# lớp cfg và lớp thực thi nằm ở HAI module khác nhau trong Isaac Lab: cfg ở commands_cfg,
# phần chạy ở velocity_command
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

__all__ = ["BalanceCarVelocityCommand", "BalanceCarVelocityCommandCfg"]


class BalanceCarVelocityCommand(UniformVelocityCommand):
    """Giống hệt bản gốc, chỉ thay phần tính metric cho đúng trục của xe này.

    Ba số được ghi log:

    * ``error_vel_fwd`` — |lệnh tiến − vận tốc tiến thật| [m/s]. **Đây là số cần theo dõi.**
      Bám hoàn hảo thì về 0; đứng im trong khi được lệnh chạy thì bằng đúng độ lớn lệnh.
    * ``error_vel_lat`` — độ lớn vận tốc NGANG [m/s]. Xe hai bánh vi sai không đi ngang được,
      nên số này khác 0 nghĩa là bánh đang trượt.
    * ``error_vel_yaw`` — |lệnh quay − tốc độ quay thật| [rad/s].
    """

    cfg: BalanceCarVelocityCommandCfg

    def __init__(self, cfg: BalanceCarVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # bản gốc đăng ký error_vel_xy; bỏ đi để không có hai số mâu thuẫn nhau trong log
        self.metrics.pop("error_vel_xy", None)
        self.metrics["error_vel_fwd"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_lat"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt

        # tách vận tốc world theo hướng tiến thật của thân thay vì lấy thẳng root_lin_vel_b:
        # làm vậy thì đổi quy ước trục trong CAD cũng không phải sửa chỗ này, chỉ sửa forward_axis
        axis = torch.zeros(self.num_envs, 3, device=self.device)
        axis[:, self.cfg.forward_axis] = self.cfg.forward_sign
        forward_w = quat_apply(self.robot.data.root_quat_w, axis)[:, :2]
        forward_w = torch.nn.functional.normalize(forward_w, dim=-1, eps=1e-6)
        left_w = torch.stack([-forward_w[:, 1], forward_w[:, 0]], dim=-1)

        vel_w = self.robot.data.root_lin_vel_w[:, :2]
        vel_fwd = torch.sum(vel_w * forward_w, dim=1)
        vel_lat = torch.sum(vel_w * left_w, dim=1)

        self.metrics["error_vel_fwd"] += torch.abs(self.vel_command_b[:, 0] - vel_fwd) / max_command_step
        self.metrics["error_vel_lat"] += torch.abs(vel_lat) / max_command_step
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_w[:, 2]) / max_command_step
        )


@configclass
class BalanceCarVelocityCommandCfg(UniformVelocityCommandCfg):
    """Cấu hình cho :class:`BalanceCarVelocityCommand`."""

    class_type: type = BalanceCarVelocityCommand

    forward_axis: int = 1
    """Trục của body ứng với hướng tiến: 0 = X, 1 = Y, 2 = Z. Xe này là **Y**."""

    forward_sign: float = 1.0
    """Dấu của trục tiến. Đổi sang -1.0 nếu bản export USD cho hướng tiến ngược lại."""
