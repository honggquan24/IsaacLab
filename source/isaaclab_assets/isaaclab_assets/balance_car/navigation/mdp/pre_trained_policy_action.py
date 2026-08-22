# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term chạy policy thăng bằng đã train làm tầng thấp.

Đây là **bản sao có chủ đích** của
``isaaclab_tasks.manager_based.navigation.mdp.PreTrainedPolicyAction``. Lý do không import
thẳng: package này nằm trong ``isaaclab_assets``, mà ``isaaclab_tasks`` lại import
``isaaclab_assets`` lúc khởi tạo — import chéo sẽ tạo vòng. Giữ bản sao nhưng **giữ đúng bản
gốc**, không sửa đổi ngoài phần ghi chú.

Cơ chế
------
Tầng cao xuất ra ``(vx, vy, wz)``. Ba số đó **không cộng vào mô-men bánh** mà được tiêm vào
đúng chỗ mà policy tầng thấp mong đợi: term quan sát ``velocity_commands`` của nó. Tầng thấp
đã được train để bám lệnh này, nên nó biết mình đang được yêu cầu làm gì và tự phối hợp
nghiêng thân với quay bánh. Cộng thẳng vào mô-men thì tầng thấp không hề biết có ai vừa đẩy
nó — nó chỉ thấy xe bị nghiêng rồi bù lại, và hai tầng chống nhau.

Vì sao ``actions`` phải được ghi đè
----------------------------------
Quan sát của tầng thấp có term ``actions`` = ``mdp.last_action``. Nếu để nguyên, hàm đó trả
về action GẦN NHẤT CỦA MÔI TRƯỜNG ĐANG CHẠY — tức action 3 chiều của tầng cao, chứ không phải
mô-men 2 chiều mà tầng thấp vừa xuất. Vector quan sát sai kích thước, và nếu tình cờ khớp
kích thước thì còn tệ hơn: sai nội dung mà không báo lỗi.

Bản trước comment hai dòng này lại, và lúc đó **là đúng** — quan sát tầng thấp của bản cũ
không hề có term ``actions``. Nhưng đó chính là cái bẫy của việc chép tay danh sách quan sát:
hôm nay thêm một term vào tầng thấp mà quên bỏ comment ở đây thì không có gì báo lỗi cả. Nay
tầng thấp dùng thẳng nhóm quan sát của nó (có ``actions`` theo đúng mẫu Isaac Lab) nên hai
dòng này là BẮT BUỘC.
"""

from __future__ import annotations

import glob
import os
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["PreTrainedBalancePolicyAction", "PreTrainedBalancePolicyActionCfg", "latest_exported_policy"]


def latest_exported_policy(experiment_name: str) -> str:
    """Đường dẫn tới ``policy.pt`` được export gần nhất của một experiment.

    Ghim cứng tên run vào config thì cứ train lại một lần là phải sửa file, và lỗi chỉ hiện ra
    lúc dựng env. Tên run là dấu thời gian nên sắp xếp chuỗi là ra bản mới nhất.

    Không tìm thấy thì trả về chính cái pattern, để thông báo lỗi của
    :class:`PreTrainedBalancePolicyAction` nói rõ nó đã tìm ở đâu.
    """
    pattern = os.path.join("logs", "rsl_rl", experiment_name, "*", "exported", "policy.pt")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else pattern


class PreTrainedBalancePolicyAction(ActionTerm):
    """Chạy policy thăng bằng đã train (TorchScript) làm tầng thấp.

    Raw action của term này là lệnh vận tốc ``(vx, vy, wz)`` cho tầng thấp.
    """

    cfg: PreTrainedBalancePolicyActionCfg

    def __init__(self, cfg: PreTrainedBalancePolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(
                f"Không tìm thấy policy tầng thấp: '{cfg.policy_path}'.\n"
                "File này do play.py sinh ra, không phải train.py. Chạy Isaac-Balance-Car một"
                " lần với play.py trước."
            )
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # tầng thấp: action mô-men bánh
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            # dọn action tầng thấp khi episode vừa reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # Hai phép ghi đè bắt buộc — xem docstring module.
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()

        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Lệnh vận tốc ``(vx, vy, wz)``."""
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        elif hasattr(self, "base_vel_goal_visualizer"):
            self.base_vel_goal_visualizer.set_visibility(False)
            self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # Mũi tên XANH LÁ = lệnh tầng cao, XANH DƯƠNG = vận tốc thật, cùng hệ thân nên so được
        # trực tiếp. Đúng hướng vì thành phần tiến của xe này nằm ở chỉ số 1 (body +Y) và
        # `atan2(v[1], v[0])` dựng góc từ cả hai thành phần.
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Đổi vận tốc XY trong hệ thân thành hướng mũi tên trong world."""
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PreTrainedBalancePolicyActionCfg(ActionTermCfg):
    """Cấu hình cho :class:`PreTrainedBalancePolicyAction`."""

    class_type: type[ActionTerm] = PreTrainedBalancePolicyAction

    asset_name: str = MISSING
    """Tên robot trong scene."""

    policy_path: str = MISSING
    """Đường dẫn tới TorchScript ``policy.pt`` của tầng thấp."""

    low_level_decimation: int = 4
    """Số bước vật lý giữa hai lần hỏi policy tầng thấp.

    **Phải bằng ``decimation`` của env tầng thấp.** ``apply_actions()`` được gọi mỗi bước vật
    lý; để lệch thì policy thăng bằng bị hỏi ở tần số khác lúc nó được train, và vận tốc/gia
    tốc nó nhìn thấy lệch hẳn so với phân phối lúc học.
    """

    low_level_actions: ActionTermCfg = MISSING
    """Action term của tầng thấp (mô-men bánh)."""

    low_level_observations: ObservationGroupCfg = MISSING
    """Nhóm quan sát của tầng thấp. Phải là **chính** nhóm ``policy`` của env tầng thấp, không
    phải bản chép tay: chép tay thì thứ tự term lệch đi lúc nào không hay và policy nhận vào
    một vector trộn sai thứ tự mà không có gì báo lỗi."""

    debug_vis: bool = True
