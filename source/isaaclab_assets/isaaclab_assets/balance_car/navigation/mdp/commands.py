# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command term sinh quỹ đạo cho xe hai bánh bám theo.

Khác với ``UniformPose2dCommand`` — bốc một điểm đích đứng yên rồi chờ xe chạy tới — term này
sinh ra một **điểm mục tiêu CHUYỂN ĐỘNG** trên một đường cong kín. Xe phải chạy đuổi liên tục,
đó mới là bám quỹ đạo.

Vì sao không dùng chuỗi waypoint rời rạc
----------------------------------------
Waypoint rời rạc cho reward dạng bậc thang: tới gần thì được thưởng một cục rồi mục tiêu nhảy
sang chỗ khác. Policy học được cách *tới điểm*, không học được cách *đi theo đường* — giữa hai
waypoint nó đi kiểu gì cũng được, và trong video thì thấy rõ là xe giật cục từng chặng. Mục
tiêu chạy trơn cho tín hiệu liên tục ở mọi thời điểm, và tốc độ của nó là thứ ràng buộc xe
phải giữ nhịp chứ không phải phóng tới rồi đứng đợi.

Quy ước hướng
-------------
Thân xe có **hướng tiến là +Y của body** (xem ``BALANCE_CAR_CFG``), không phải +X như quy ước
locomotion của Isaac Lab. Lệnh ở đây được đưa về **hệ heading** — thành phần dọc hướng tiến và
thành phần ngang — nên nó độc lập với quy ước trục, đổi CAD cũng không phải sửa gì ở đây.

Action mà tầng cao xuất ra là ``[vx, vy, wz]`` của tầng thấp, trong đó **``vy`` mới là lệnh
tiến**. Policy tầng cao tự học ánh xạ đó nên không cần đổi chỗ ở code, nhưng đọc log thì phải
nhớ: cột action thứ hai là ga, không phải cột thứ nhất.

Vector lệnh, shape ``(num_envs, 4)``
------------------------------------
====  ==========================================================================
chỉ số  ý nghĩa
====  ==========================================================================
0     sai số DỌC hướng tiến [m] — dương là mục tiêu ở phía trước
1     sai số NGANG [m] — dương là mục tiêu ở bên trái
2     sai số HƯỚNG [rad] — góc giữa hướng tiến của xe và tiếp tuyến quỹ đạo
3     tốc độ mục tiêu chạy trên quỹ đạo [m/s]
====  ==========================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["PathCommand", "PathCommandCfg"]

PATH_TYPES = ("circle", "figure8")
"""Các dạng quỹ đạo hỗ trợ. Chỉ số trong tuple này là mã lưu trong ``self._path_type``."""


class PathCommand(CommandTerm):
    """Sinh một điểm mục tiêu chạy trên đường cong kín để xe bám theo.

    Mỗi env bốc riêng dạng đường, bán kính, tốc độ và chiều chạy, nên policy không học thuộc
    được một quỹ đạo cụ thể.

    Đường luôn được đặt sao cho **điểm xuất phát của mục tiêu trùng vị trí xe lúc reset**. Nếu
    không, xe sinh ra đã cách quỹ đạo vài mét và phần lớn episode chỉ là chạy tới đường chứ
    không phải bám nó — tín hiệu học bị loãng đúng ở chỗ quan trọng nhất.
    """

    cfg: PathCommandCfg

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # lệnh trả cho policy
        self.path_command = torch.zeros(self.num_envs, 4, device=self.device)

        # tham số đường, mỗi env một bộ
        self._center = torch.zeros(self.num_envs, 2, device=self.device)
        self._radius = torch.zeros(self.num_envs, device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)
        self._spin = torch.ones(self.num_envs, device=self.device)  # +1 hoặc -1
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._path_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # mã của các dạng đường được phép bốc
        self._allowed = torch.tensor(
            [PATH_TYPES.index(name) for name in cfg.path_types], dtype=torch.long, device=self.device
        )

        self.metrics["path_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PathCommand:\n"
        msg += f"\tDạng đường     : {self.cfg.path_types}\n"
        msg += f"\tBán kính       : {self.cfg.radius_range} m\n"
        msg += f"\tTốc độ mục tiêu: {self.cfg.speed_range} m/s\n"
        msg += f"\tChu kỳ đổi đường: {self.cfg.resampling_time_range} s"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """``(num_envs, 4)`` — xem bảng ở docstring của module."""
        return self.path_command

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        sample = torch.empty(n, device=self.device)

        self._radius[env_ids] = sample.uniform_(*self.cfg.radius_range).clone()
        self._speed[env_ids] = sample.uniform_(*self.cfg.speed_range).clone()
        self._spin[env_ids] = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            torch.ones(n, device=self.device),
            -torch.ones(n, device=self.device),
        )
        self._path_type[env_ids] = self._allowed[torch.randint(len(self._allowed), (n,), device=self.device)]
        self._phase[env_ids] = 0.0

        # đặt tâm sao cho điểm pha 0 rơi đúng vào chỗ xe đang đứng
        offset = self._shape(env_ids, self._phase[env_ids])
        self._center[env_ids] = self.robot.data.root_pos_w[env_ids, :2] - offset

    def _update_command(self):
        # tốc độ pha suy từ tốc độ dài mong muốn: với đường tròn thì đúng tuyệt đối
        # (v = ω·R), với figure-8 là xấp xỉ — đủ dùng vì reward chấm theo VỊ TRÍ mục tiêu,
        # không chấm theo tốc độ.
        self._phase += (self._speed / self._radius) * self._spin * self._env.step_dt

        env_ids = slice(None)
        target_w = self._center + self._shape(env_ids, self._phase)
        tangent_w = self._tangent(env_ids, self._phase)

        forward_w, left_w = self._heading_frame()
        delta_w = target_w - self.robot.data.root_pos_w[:, :2]

        self.path_command[:, 0] = torch.sum(delta_w * forward_w, dim=1)
        self.path_command[:, 1] = torch.sum(delta_w * left_w, dim=1)
        # góc giữa hướng tiến của xe và tiếp tuyến, lấy qua atan2 trong hệ heading nên
        # tự nằm trong [-pi, pi] và không có điểm gãy
        self.path_command[:, 2] = torch.atan2(
            torch.sum(tangent_w * left_w, dim=1),
            torch.sum(tangent_w * forward_w, dim=1),
        )
        self.path_command[:, 3] = self._speed

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["path_error"] += torch.norm(self.path_command[:, :2], dim=1) / max_command_step
        self.metrics["heading_error"] += torch.abs(self.path_command[:, 2]) / max_command_step

    """
    Helper functions.
    """

    def _shape(self, env_ids, phase: torch.Tensor) -> torch.Tensor:
        """Điểm trên đường (so với tâm) tại pha đã cho. Shape ``(n, 2)``."""
        radius = self._radius[env_ids]
        spin = self._spin[env_ids]
        is_circle = self._path_type[env_ids] == PATH_TYPES.index("circle")

        circle = torch.stack([radius * torch.cos(phase), radius * spin * torch.sin(phase)], dim=-1)
        # lemniscate kiểu Gerono: chạy hình số 8 nằm ngang, đi qua tâm hai lần mỗi vòng
        fig8 = torch.stack([radius * torch.sin(phase), radius * spin * torch.sin(2.0 * phase) * 0.5], dim=-1)
        return torch.where(is_circle.unsqueeze(-1), circle, fig8)

    def _tangent(self, env_ids, phase: torch.Tensor) -> torch.Tensor:
        """Vector tiếp tuyến ĐƠN VỊ theo chiều chạy. Shape ``(n, 2)``."""
        spin = self._spin[env_ids]
        is_circle = self._path_type[env_ids] == PATH_TYPES.index("circle")

        circle = torch.stack([-torch.sin(phase), spin * torch.cos(phase)], dim=-1)
        fig8 = torch.stack([torch.cos(phase), spin * torch.cos(2.0 * phase)], dim=-1)
        tangent = torch.where(is_circle.unsqueeze(-1), circle, fig8)
        # nhân thêm chiều chạy: pha đi lùi thì tiếp tuyến cũng đảo
        tangent = tangent * spin.unsqueeze(-1)
        return torch.nn.functional.normalize(tangent, dim=-1, eps=1e-6)

    def _heading_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Hai vector đơn vị trong mặt phẳng ngang: hướng tiến và hướng bên trái của xe.

        Hướng tiến của xe này là **+Y của body**, không phải +X — xem docstring module.
        """
        axis = torch.zeros(self.num_envs, 3, device=self.device)
        axis[:, 1] = 1.0
        forward = quat_apply(self.robot.data.root_quat_w, axis)[:, :2]
        forward = torch.nn.functional.normalize(forward, dim=-1, eps=1e-6)
        # quay +90° quanh trục z: (x, y) -> (-y, x)
        left = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)
        return forward, left

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
                self.path_visualizer = VisualizationMarkers(self.cfg.path_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.path_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            self.path_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        z = self.robot.data.root_pos_w[:, 2:3]

        target_w = self._center + self._shape(slice(None), self._phase)
        self.goal_visualizer.visualize(translations=torch.cat([target_w, z], dim=1))

        # rải đều điểm quanh một vòng để thấy nguyên hình quỹ đạo trong video
        steps = torch.linspace(0.0, 2.0 * torch.pi, self.cfg.path_vis_samples, device=self.device)
        phases = self._phase.unsqueeze(1) * 0.0 + steps.unsqueeze(0)  # (num_envs, S)
        pts = []
        for k in range(self.cfg.path_vis_samples):
            pts.append(self._center + self._shape(slice(None), phases[:, k]))
        path_w = torch.stack(pts, dim=1).reshape(-1, 2)
        z_rep = z.repeat_interleave(self.cfg.path_vis_samples, dim=0)
        self.path_visualizer.visualize(translations=torch.cat([path_w, z_rep], dim=1))


@configclass
class PathCommandCfg(CommandTermCfg):
    """Cấu hình cho :class:`PathCommand`."""

    class_type: type = PathCommand

    asset_name: str = "robot"
    """Tên robot trong scene."""

    path_types: tuple[str, ...] = PATH_TYPES
    """Các dạng đường được bốc. Xem :data:`PATH_TYPES`."""

    radius_range: tuple[float, float] = (1.0, 2.0)
    """Bán kính đường [m]. Nhỏ quá thì xe hai bánh vi sai phải quay gắt liên tục."""

    speed_range: tuple[float, float] = (0.4, 0.9)
    """Tốc độ điểm mục tiêu chạy trên đường [m/s].

    Phải nằm TRONG dải lệnh của tầng thấp (``lin_vel_y`` = ±1.5 m/s — thành phần TIẾN của xe
    này nằm ở trục Y, xem ``balance_env_cfg``). Đặt cao hơn thì mục tiêu
    chạy nhanh hơn khả năng bám của tầng thấp, xe không bao giờ đuổi kịp và tín hiệu học chỉ
    còn là "luôn luôn tụt lại" — không phân biệt được policy tốt với policy dở.
    """

    path_vis_samples: int = 32
    """Số điểm rải trên đường khi bật ``debug_vis``. Chỉ ảnh hưởng hiển thị."""

    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/path_goal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.09,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.15)),
            ),
        },
    )
    """Quả cầu đỏ = điểm mục tiêu đang chạy."""

    path_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/path_line",
        markers={
            "path": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.6, 1.0)),
            ),
        },
    )
    """Chuỗi chấm xanh = nguyên hình quỹ đạo, để video nhìn ra xe đang bám cái gì."""
