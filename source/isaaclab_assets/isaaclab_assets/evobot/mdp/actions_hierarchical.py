# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-trained policy action term for evobot navigation."""

from __future__ import annotations

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


class PreTrainedBalancePolicyAction(ActionTerm):
    """Pre-trained balance policy action term for evobot_v1.

    This action term uses a pre-trained balance policy trained with velocity commands.
    The high-level policy outputs velocity commands (vx, vy, omega) which are passed
    to the low-level policy via observation remapping.

    IMPORTANT: Low-level policy output actions are in NORMALIZED range [-1, 1].
    The ActionTerm.process_actions() will scale them by the configured scale factors.
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
        self._raw_actions_ = torch.zeros(self.num_envs, 7, device=self.device)

        # Prepare low level actions (joint velocity)
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)

        # IMPORTANT: Low-level policy outputs NORMALIZED actions [-1, 1]
        # These will be scaled by low_level_action_term.process_actions()
        self.low_level_actions = torch.zeros(self.num_envs, 5, device=self.device)

        # Store last low level actions for observation
        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        print(f"cfg.low_level_observations: {cfg.low_level_observations}")

        # Remap observations for low level policy
        # Low-level policy will receive high-level velocity commands as base_velocity_cmd
        cfg.low_level_observations.last_action.func = lambda _: last_action()
        cfg.low_level_observations.last_action.params = {}

        cfg.low_level_observations.base_velocity_cmd.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.base_velocity_cmd.params = dict()

        cfg.low_level_observations.arm_ee_pose_cmd.func = lambda dummy_env: self._raw_actions_
        cfg.low_level_observations.arm_ee_pose_cmd.params = dict()

        cfg.low_level_observations.grip_ee_pose_left_cmd.func = lambda dummy_env: self._raw_actions_
        cfg.low_level_observations.grip_ee_pose_left_cmd.params = dict()

        cfg.low_level_observations.grip_ee_pose_right_cmd.func = lambda dummy_env: self._raw_actions_
        cfg.low_level_observations.grip_ee_pose_right_cmd.params = dict()

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
        """Process high-level velocity command actions.

        High-level policy outputs velocity commands which should be in range [-1, 1].
        We clip them to ensure they stay within bounds before passing to low-level policy.
        """
        # Clip high-level actions to normalized range [-1, 1]
        # This prevents exploration noise from creating invalid velocity commands
        self._raw_actions[:] = torch.clamp(actions, min=-1.0, max=1.0)

    def apply_actions(self):
        """Apply actions by running low-level policy with velocity commands.

        Strategy:
        - High-level policy outputs velocity commands [vx, vy, omega] in range [-1, 1]
        - These are passed to low-level policy via base_velocity_cmd observation
        - Low-level policy outputs NORMALIZED joint actions [-1, 1]
        - Low-level ActionTerm scales these by configured scale factors (e.g., 200.0)
        - Result: smooth velocity tracking with proper balance
        """
        if self._counter % self.cfg.low_level_decimation == 0:
            # IMPORTANT: self._raw_actions is already remapped to low_level_observations.base_velocity_cmd
            # via the lambda function in __init__ (line 78-79)
            # So the low-level policy will see self._raw_actions as velocity command!

            # Get observations for low-level policy (includes velocity commands from high-level)
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # Run low-level policy to get NORMALIZED joint actions
            # Low-level policy sees:
            # - IMU, joint states (actual robot state)
            # - base_velocity_cmd = self._raw_actions (from high-level policy)
            # - arm/gripper commands = zeros (not used)
            # Output: NORMALIZED [left_wheel, right_wheel, arm, left_grip, right_grip] in [-1, 1]
            low_level_actions = self.policy(low_level_obs)

            # Use low-level policy output directly (already normalized)
            self.low_level_actions[:] = low_level_actions

            # Optional: Add velocity modulation on top if needed
            # This is only necessary if low-level doesn't track velocity well
            if self.cfg.velocity_scale != 0.0 or self.cfg.turn_scale != 0.0:
                vx = self._raw_actions[:, 0:1]  # Forward velocity
                omega = self._raw_actions[:, 2:3]  # Angular velocity

                vel_scale = self.cfg.velocity_scale
                turn_scale = self.cfg.turn_scale

                # Add velocity modulation to wheel actions (still in normalized range)
                self.low_level_actions[:, 0:1] += vx * vel_scale - omega * turn_scale  # Left wheel
                self.low_level_actions[:, 1:2] += vx * vel_scale + omega * turn_scale  # Right wheel

                # Clip to normalized range [-1, 1]
                self.low_level_actions[:] = torch.clamp(self.low_level_actions, min=-1.0, max=1.0)

            # Low-level ActionTerm will scale these normalized actions by configured scale factors
            # e.g., wheels: 200.0, arm: 30.0, grippers: 30.0
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

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

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
    """Low level action configuration (joint velocity)."""

    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration for balance policy."""

    velocity_scale: float = 0.0
    """Additional scale factor for forward velocity commands (on top of low-level policy output).
    Set to 0.0 to trust low-level policy completely."""

    turn_scale: float = 0.0
    """Additional scale factor for turning commands (on top of low-level policy output).
    Set to 0.0 to trust low-level policy completely."""

    debug_vis: bool = True
    """Whether to visualize debug information."""
