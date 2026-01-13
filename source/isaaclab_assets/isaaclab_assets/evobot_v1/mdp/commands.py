# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command generators for Evobot V1."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformJointPositionCommand(CommandTerm):
    """Command generator for uniform sampling of 1D joint position targets.

    This command generates random target positions for a single joint (e.g., arm_joint).
    Useful for manipulation tasks where you want the robot to reach different arm heights.
    """

    cfg: UniformJointPositionCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformJointPositionCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration object.
            env: The environment.
        """
        super().__init__(cfg, env)

        # Get robot articulation
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Store joint index for the commanded joint
        self.joint_ids = self.robot.find_joints(cfg.joint_name)[0]

        # Create buffers for command: [num_envs, 1] - target joint position
        self.command = torch.zeros(self.num_envs, 1, device=self.device)

        # Track time for resampling
        self.time_left = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformJointPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tJoint name: {self.cfg.joint_name}\n"
        return msg

    @property
    def command_dim(self) -> int:
        """Dimension of the command: 1 (joint position)."""
        return 1

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call.
        """
        # Decrease time left
        self.time_left -= dt

        # Resample commands for envs where time_left <= 0
        env_ids = (self.time_left <= 0).nonzero(as_tuple=False).flatten()
        self._resample(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command for given environment indices.

        Args:
            env_ids: The environment indices to reset. If None, reset all.

        Returns:
            An empty dictionary (no logging metrics).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Resample commands
        self._resample(env_ids)

        return {}

    def _resample(self, env_ids: torch.Tensor):
        """Resample commands for given environment indices.

        Args:
            env_ids: The environment indices to resample commands for.
        """
        # Sample new joint position targets uniformly
        r = torch.empty(len(env_ids), device=self.device)
        self.command[env_ids, 0] = r.uniform_(*self.cfg.ranges.joint_pos)

        # Reset time left for resampling
        r = torch.empty(len(env_ids), device=self.device)
        self.time_left[env_ids] = r.uniform_(*self.cfg.resampling_time_range)


@configclass
class UniformJointPositionCommandCfg(CommandTermCfg):
    """Configuration for uniform joint position command generator."""

    class_type: type = UniformJointPositionCommand
    """The associated command term class."""

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    joint_name: str = "arm_joint"
    """Name of the joint to command (e.g., 'arm_joint')."""

    resampling_time_range: tuple[float, float] = (5.0, 10.0)
    """Time range for resampling commands (min, max) in seconds."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for joint position command."""

        joint_pos: tuple[float, float] = (-1.57, 1.57)
        """Range for target joint position (radians). Default: -90° to +90°."""

    ranges: Ranges = Ranges()
    """Distribution ranges for the command."""


class BinaryGripperCommand(CommandTerm):
    """Command generator for binary gripper control (open/close).

    This command generates binary commands for gripper: 0 = open, 1 = close.
    The target position is normalized between 0 and 1 for easy interpretation.
    """

    cfg: BinaryGripperCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BinaryGripperCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration object.
            env: The environment.
        """
        super().__init__(cfg, env)

        # Get robot articulation
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Create buffers for command: [num_envs, 1] - binary gripper state
        # 0.0 = open, 1.0 = close
        self.command = torch.zeros(self.num_envs, 1, device=self.device)

        # Track time for resampling
        self.time_left = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "BinaryGripperCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tOpen probability: {self.cfg.open_prob}\n"
        return msg

    @property
    def command_dim(self) -> int:
        """Dimension of the command: 1 (binary state)."""
        return 1

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call.
        """
        # Decrease time left
        self.time_left -= dt

        # Resample commands for envs where time_left <= 0
        env_ids = (self.time_left <= 0).nonzero(as_tuple=False).flatten()
        self._resample(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command for given environment indices.

        Args:
            env_ids: The environment indices to reset. If None, reset all.

        Returns:
            An empty dictionary (no logging metrics).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Resample commands
        self._resample(env_ids)

        return {}

    def _resample(self, env_ids: torch.Tensor):
        """Resample commands for given environment indices.

        Args:
            env_ids: The environment indices to resample commands for.
        """
        # Sample binary gripper state: 0 (open) or 1 (close)
        # Use Bernoulli distribution with open_prob
        r = torch.rand(len(env_ids), device=self.device)
        self.command[env_ids, 0] = (r > self.cfg.open_prob).float()

        # Reset time left for resampling
        r = torch.empty(len(env_ids), device=self.device)
        self.time_left[env_ids] = r.uniform_(*self.cfg.resampling_time_range)


@configclass
class BinaryGripperCommandCfg(CommandTermCfg):
    """Configuration for binary gripper command generator."""

    class_type: type = BinaryGripperCommand
    """The associated command term class."""

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    resampling_time_range: tuple[float, float] = (3.0, 5.0)
    """Time range for resampling commands (min, max) in seconds."""

    open_prob: float = 0.5
    """Probability of sampling 'open' command (default: 0.5 for balanced distribution)."""
