# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-trained low-level locomotion policy as a high-level action term (V5).

Khác bản V3: low-level V5 có NHIỀU action term (leg_pos + wheel_vel) nên dùng
`ActionManager` thay vì một `ActionTerm` đơn. Obs remap theo tên term của V5
(`last_action`, `velocity_cmd`).

Luồng:
  - High-level policy xuất raw_actions = velocity command (vx, vy, omega).
  - Mỗi `low_level_decimation` bước: dựng obs low-level (trong đó velocity_cmd =
    raw_actions), chạy policy đã train → action khớp, áp qua ActionManager.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import (
    ActionManager,
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationManager,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedPolicyAction(ActionTerm):
    """Chạy policy locomotion V5 đã train làm tầng thấp. Raw action = (vx, vy, omega)."""

    cfg: PreTrainedPolicyActionCfg

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' không tồn tại.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # dải lệnh (vx, vy, wz) — nhân với output [-1,1] của tầng cao
        self._command_scale = torch.tensor(cfg.command_scale, device=self.device)

        # Low-level dùng ActionManager (V5: leg_pos mimic/nomimic + wheel_vel)
        self._low_level_action_manager = ActionManager(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(
            self.num_envs, self._low_level_action_manager.total_action_dim, device=self.device
        )

        def last_action():
            # reset last action về 0 ở bước đầu mỗi episode
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # Remap obs low-level (tên term theo V5 PolicyCfg):
        #   last_action  -> action low-level vừa sinh
        #   velocity_cmd -> raw action tầng cao (lệnh vận tốc)
        cfg.low_level_observations.last_action.func = lambda dummy_env: last_action()
        cfg.low_level_observations.last_action.params = dict()
        cfg.low_level_observations.velocity_cmd.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_cmd.params = dict()

        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        self._counter = 0

    @property
    def action_dim(self) -> int:
        return 3  # (vx, vy, omega) — khớp velocity_cmd 3 chiều của low-level

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        # Tầng cao xuất [-1,1]^3, nhân command_scale = dải lệnh thực tế (vx,vy,wz).
        # vx_scale=0 → ép vx=0 (robot không đi ngang); vy/wz_scale = tốc/góc tối đa.
        # Đặt range bằng cách sửa `command_scale` trong PreTrainedPolicyActionCfg.
        self._raw_actions[:] = actions.clamp(-1.0, 1.0) * self._command_scale

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_manager.process_action(self.low_level_actions)
            self._counter = 0
        self._low_level_action_manager.apply_action()
        self._counter += 1

    # ───────────────────────── debug visualization ──────────────────────────
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
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Config cho action term locomotion-pretrained."""

    class_type: type[ActionTerm] = PreTrainedPolicyAction

    asset_name: str = MISSING
    policy_path: str = MISSING  # đường dẫn policy.pt đã export (low-level)
    low_level_decimation: int = 4  # low-level chạy nhanh gấp 4 lần high-level
    low_level_actions: object = MISSING  # ActionCfg group của locomotion (leg+wheel)
    low_level_observations: ObservationGroupCfg = MISSING
    # Dải lệnh vận tốc đưa vào low-level = (vx_max, vy_max, wz_max). Tầng cao xuất
    # [-1,1]^3 rồi nhân các hệ số này. vx_max=0 → robot không đi ngang (khớp lúc
    # train low-level). vy/wz_max nên ≤ dải train locomotion (±0.5).
    command_scale: tuple[float, float, float] = (0.0, 0.5, 0.5)
    debug_vis: bool = True
