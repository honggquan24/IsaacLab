# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-trained policy action term for balance car navigation."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedBalancePolicyAction(ActionTerm):
    """Pre-trained balance policy action term for cart_v1.

    This action term uses a pre-trained balance policy and adds velocity commands
    for navigation. The high-level actions are velocity commands (vx, vy, omega)
    that are added to the balance policy's actions.
    """

    cfg: PreTrainedBalancePolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedBalancePolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Load pre-trained balance policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # Raw actions are velocity commands (vx, vy, omega) for navigation
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # Prepare low level actions (joint effort)
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # Store last low level actions for observation
        def last_action():
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # Remap observations for low level policy
        # The balance policy expects observations without velocity commands
        # We need to provide: joint_pos, joint_vel, pitch, roll, yaw, lin_vel, ang_vel
        # cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        # cfg.low_level_observations.actions.params = dict()

        # Create observation manager for low level policy
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    @property
    def action_dim(self) -> int:
        """Dimension of high-level actions (velocity commands: vx, vy, omega)."""
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor):
        """Process high-level velocity command actions."""
        self._raw_actions[:] = actions

    def apply_actions(self):
        """Apply actions by running low-level balance policy with velocity modulation."""
        if self._counter % self.cfg.low_level_decimation == 0:
            # Get observations for low-level policy
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # Run balance policy to get base wheel efforts
            balance_actions = self.policy(low_level_obs)

            # Modulate actions based on velocity commands
            # raw_actions: [vx, vy, omega] where:
            # - vx: forward velocity command
            # - vy: lateral velocity (not used for diff drive)
            # - omega: angular velocity (turning)
            vx = self._raw_actions[:, 0:1]  # Forward velocity
            omega = self._raw_actions[:, 2:3]  # Angular velocity

            # Differential drive: left wheel, right wheel
            # Forward motion: both wheels same direction
            # Turning: wheels opposite direction
            vel_scale = self.cfg.velocity_scale
            turn_scale = self.cfg.turn_scale

            # Add velocity commands to balance actions
            # Left wheel: +forward, -turn (for positive omega = turn left)
            # Right wheel: +forward, +turn
            velocity_modulation = torch.zeros_like(balance_actions)
            velocity_modulation[:, 0:1] = vx * vel_scale - omega * turn_scale  # Left wheel
            velocity_modulation[:, 1:2] = vx * vel_scale + omega * turn_scale  # Right wheel

            # Combine balance policy output with velocity modulation
            self.low_level_actions[:] = balance_actions + velocity_modulation

            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
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
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts XY velocity to arrow visualization."""
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Use only yaw component to avoid arrow pointing down due to robot's roll/pitch
        base_yaw_quat = math_utils.yaw_quat(self.robot.data.root_quat_w)
        arrow_quat = math_utils.quat_mul(base_yaw_quat, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class PreTrainedBalancePolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained balance policy action term."""

    class_type: type[ActionTerm] = PreTrainedBalancePolicyAction
    """Class of the action term."""

    asset_name: str = MISSING
    """Name of the asset in the environment."""

    policy_path: str = MISSING
    """Path to the pre-trained balance policy (.pt file)."""

    low_level_decimation: int = 1
    """Decimation factor for the low level action term."""

    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration (joint effort)."""

    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration for balance policy."""

    velocity_scale: float = 10.0
    """Scale factor for forward velocity commands."""

    turn_scale: float = 5.0
    """Scale factor for turning commands."""

    debug_vis: bool = True
    """Whether to visualize debug information."""
