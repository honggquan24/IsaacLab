# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose2d command tùy biến cho marker đích (chấm tròn).

Khác UniformPose2dCommand gốc ở phần HIỂN THỊ debug:
  - Đặt marker ở độ cao `marker_height` (ngang tầm thân robot) thay vì z mặc định.
  - ẨN marker (đẩy xuống dưới đất) khi robot vào trong bán kính `hide_radius`
    — tạo hiệu ứng "đích biến mất khi đã tới".
Logic sinh đích / reward / metric giữ nguyên như lớp gốc.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz


class HideOnReachPose2dCommand(UniformPose2dCommand):
    """Như UniformPose2dCommand nhưng:
    - marker đặt ở tầm robot, ẩn khi robot vào trong `hide_radius`;
    - nếu `resample_on_reach`: robot tới trong `reach_radius` → sinh ĐÍCH MỚI
      ngay (điều hướng liên tục), không chờ hết thời gian resample.
    """

    cfg: HideOnReachPose2dCommandCfg

    def _update_command(self):
        super()._update_command()
        if self.cfg.resample_on_reach:
            dist = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
            reached = (dist < self.cfg.reach_radius).nonzero(as_tuple=False).flatten()
            if len(reached) > 0:
                self._resample(reached)  # đổi đích + reset timer cho các env đã tới

    def _debug_vis_callback(self, event):
        # toạ độ marker: x,y của đích, z = tầm thân robot
        trans = self.pos_command_w.clone()
        trans[:, 2] = self.cfg.marker_height

        # ẩn marker cho env đã tới (đẩy xuống dưới đất, khuất tầm nhìn)
        dist = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        trans[dist < self.cfg.hide_radius, 2] = -1000.0

        self.goal_pose_visualizer.visualize(
            translations=trans,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )


@configclass
class HideOnReachPose2dCommandCfg(UniformPose2dCommandCfg):
    class_type: type = HideOnReachPose2dCommand

    marker_height: float = 0.35  # m — chấm tròn ngang tầm thân robot

    resample_on_reach: bool = True  # tới đích → sinh đích mới ngay (chấm cũ biến mất, chấm mới hiện chỗ khác)
    reach_radius: float = 0.3  # m — coi là "đã tới" (khớp threshold goal_reached)
    hide_radius: float = 0.3  # m — = reach_radius cho nhất quán (dùng khi tắt resample_on_reach)
