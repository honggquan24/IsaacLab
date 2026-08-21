# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command term sinh vị trí mục tiêu cho xe đẩy chạy trên ray."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["CartPositionCommand", "CartPositionCommandCfg"]


class CartPositionCommand(CommandTerm):
    """Sinh vị trí mục tiêu cho khớp trượt của xe đẩy.

    Lệnh là một số vô hướng: vị trí mong muốn của xe trên ray, tính theo toạ độ khớp
    ``Slider_1`` (mét). Cứ sau ``resampling_time_range`` giây thì bốc lại một mục tiêu mới,
    nên trong video robot sẽ chạy qua chạy lại giữa các mốc thay vì đứng yên một chỗ.

    Khoảng lấy mẫu mặc định bám theo giới hạn khớp có thật trong USD
    (:attr:`~CartPositionCommandCfg.limit_ratio` × giới hạn mềm), nên không cần biết trước
    ray dài bao nhiêu; đặt :attr:`~CartPositionCommandCfg.position_range` nếu muốn chỉ định tay.
    """

    cfg: CartPositionCommandCfg
    """Cấu hình của command term."""

    def __init__(self, cfg: CartPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # lấy robot và chỉ số khớp trượt
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_idx = self.robot.find_joints(cfg.joint_name)[0][0]

        # bộ đệm lệnh: (num_envs, 1)
        self.pos_command = torch.zeros(self.num_envs, 1, device=self.device)
        # khoảng lấy mẫu, giải sau lần reset đầu tiên (lúc đó giới hạn khớp mới có dữ liệu)
        self._sample_range: tuple[float, float] | None = None

        # hằng số dùng cho marker
        self._rail_axis = torch.tensor(cfg.rail_axis, device=self.device)
        self._marker_offset = torch.tensor(cfg.marker_offset, device=self.device)

        # số liệu để ghi log
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "CartPositionCommand:\n"
        msg += f"\tKhớp điều khiển: {self.cfg.joint_name}\n"
        msg += f"\tChu kỳ đổi mục tiêu: {self.cfg.resampling_time_range} s\n"
        msg += f"\tKhoảng lấy mẫu: {self._sample_range if self._sample_range else 'theo giới hạn khớp'}"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Vị trí xe đẩy mong muốn. Shape là (num_envs, 1)."""
        return self.pos_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # sai số trung bình trong một chu kỳ lệnh (cùng cách tính với command term của Isaac Lab)
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        error = torch.abs(self.pos_command[:, 0] - self.robot.data.joint_pos[:, self.joint_idx])
        self.metrics["position_error"] += error / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        lower, upper = self._resolve_sample_range()
        self.pos_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(lower, upper)

    def _update_command(self):
        # mục tiêu đứng yên giữa hai lần lấy mẫu
        pass

    """
    Helper functions.
    """

    def _resolve_sample_range(self) -> tuple[float, float]:
        """Trả về khoảng lấy mẫu, tính một lần rồi nhớ lại."""
        if self._sample_range is None:
            if self.cfg.position_range is not None:
                self._sample_range = self.cfg.position_range
            else:
                limits = self.robot.data.soft_joint_pos_limits[0, self.joint_idx]
                self._sample_range = (
                    float(limits[0]) * self.cfg.limit_ratio,
                    float(limits[1]) * self.cfg.limit_ratio,
                )
        return self._sample_range

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # robot có thể chưa khởi tạo xong (hoặc đã bị huỷ) khi callback chạy
        if not self.robot.is_initialized:
            return
        # mốc mục tiêu nằm trên ray: gốc thân robot + trục ray × vị trí mong muốn
        target_pos_w = self.robot.data.root_pos_w + self._rail_axis * self.pos_command + self._marker_offset
        self.goal_visualizer.visualize(translations=target_pos_w)


@configclass
class CartPositionCommandCfg(CommandTermCfg):
    """Cấu hình cho :class:`CartPositionCommand`."""

    class_type: type = CartPositionCommand

    asset_name: str = "robot"
    """Tên robot trong scene."""

    joint_name: str = "Slider_1"
    """Tên khớp trượt của xe đẩy."""

    position_range: tuple[float, float] | None = None
    """Khoảng vị trí mục tiêu [m]. Để ``None`` thì suy ra từ giới hạn khớp × :attr:`limit_ratio`."""

    limit_ratio: float = 0.6
    """Phần giới hạn khớp được dùng khi :attr:`position_range` là ``None``.

    Chừa lại biên ray để xe còn chỗ giảm tốc thay vì đâm vào đầu ray.
    """

    rail_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Hướng của ray trong hệ world, chỉ dùng để vẽ marker."""

    marker_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Dịch marker so với gốc thân robot, dùng khi xe ở vị trí 0 không trùng gốc thân."""

    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cart_position_goal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
        },
    )
    """Marker đánh dấu vị trí mục tiêu trên ray."""
