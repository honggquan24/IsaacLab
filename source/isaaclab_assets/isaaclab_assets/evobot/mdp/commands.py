# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms riêng của evoBOT.

``BinaryGripperCommand`` giống ``UniformPoseCommand`` nhưng thành phần ``pos_z``
là nhị phân (0.0 = đóng kẹp, 1.0 = mở kẹp) thay vì liên tục, để policy học hành
vi đóng/mở rời rạc mà vẫn giữ nguyên dạng quan sát pose 7 chiều.

Trước đây term này được vá thẳng vào ``isaaclab.envs.mdp.commands``; nay đặt tại
package của dự án để không phải fork Isaac Lab core.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BinaryGripperCommand(CommandTerm):
    """Command generator for binary gripper control.

    Similar to UniformPoseCommand, but pos_z is binary (0.0 or 1.0) instead of continuous.
    All other components (pos_x, pos_y, roll, pitch, yaw) are sampled uniformly like UniformPoseCommand.

    This allows the gripper to learn discrete open/close behaviors while maintaining
    compatibility with existing pose command observations.
    """

    cfg: BinaryGripperCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BinaryGripperCommandCfg, env: ManagerBasedEnv):
        """Initialize the binary gripper command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]
        # obtain the body index for which the command is generated
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers for the pose command: (num_envs, 7) - [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        # -- base frame (for policy observation)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0  # Initialize quaternion to identity (w=1)
        # -- world frame (for visualization and metrics)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.pose_command_w[:, 3] = 1.0

        # metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gripper_binary_state"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "BinaryGripperCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tBinary probability (open): {self.cfg.prob_open}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command with binary z. Shape is (num_envs, 7).

        The first three elements correspond to the position [x, y, z_binary],
        followed by the quaternion orientation in (w, x, y, z).

        Note: pos_z is binary (0.0 = close, 1.0 = open)
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for logging and visualization."""
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        self.metrics["gripper_binary_state"] = self.pose_command_b[:, 2]  # z is binary state

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample pose commands for specified environments.

        Similar to UniformPoseCommand, but pos_z is sampled as binary (0.0 or 1.0).

        Args:
            env_ids: Environment IDs to resample commands for.
        """
        # sample new pose targets
        # -- position (x, y sampled uniformly, z is BINARY)
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)

        # BINARY z: sample 0.0 or 1.0 based on prob_open
        binary_z = torch.rand(len(env_ids), device=self.device) < self.cfg.prob_open
        self.pose_command_b[env_ids, 2] = binary_z.float()  # 0.0 = close, 1.0 = open

        # # Debug: Print binary command values
        # if len(env_ids) > 0 and env_ids[0] == 0:  # Only print for env 0
        # print(f"[BinaryGripperCommand] Resampled z for {self.cfg.body_name}: {self.pose_command_b[0,
        # 2].item():.1f}")

        # -- orientation (sampled uniformly like UniformPoseCommand)
        euler_angles = torch.zeros(len(env_ids), 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

        # make sure the quaternion has real part as positive (if configured)
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        """Update the command (no transformation needed - handled in _update_metrics)."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


@configclass
class BinaryGripperCommandCfg(CommandTermCfg):
    """Configuration for binary gripper command generator.

    Similar to UniformPoseCommandCfg, but pos_z is binary (0.0 or 1.0) instead of continuous.
    All other pose components (x, y, roll, pitch, yaw) are sampled uniformly like UniformPoseCommand.
    """

    class_type: type = BinaryGripperCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False."""

    @configclass
    class Ranges:
        """Distribution ranges for the pose commands (same as UniformPoseCommand)."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        # NOTE: pos_z is IGNORED for binary command - always outputs 0.0 or 1.0

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the pose commands."""

    prob_open: float = 0.5
    """Probability of generating 'open' command (z=1.0). Default is 0.5 (equal probability for 0.0 and 1.0)."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
