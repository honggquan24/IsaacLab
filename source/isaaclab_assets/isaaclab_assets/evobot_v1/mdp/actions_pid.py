# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PID-based action manager for evobot velocity control."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..controllers.pid_controller import VelocityPIDController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VelocityPIDActionTerm(ActionTerm):
    """Action term that uses PID controller to track velocity commands.

    This action term receives high-level velocity commands (vx, wz) and uses
    PID controllers to compute wheel velocities. The robot's current velocity
    is used as feedback for closed-loop control.

    The action space is (vx_cmd, wz_cmd) and internally converts to 5 DOF actions:
    [left_wheel_vel, right_wheel_vel, arm_vel, left_grip_vel, right_grip_vel]
    """

    cfg: VelocityPIDActionTermCfg
    """Configuration for the action term."""

    def __init__(self, cfg: VelocityPIDActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Get robot asset
        self._robot = env.scene[cfg.asset_name]

        # Create PID controller
        self._pid_controller = VelocityPIDController(
            kp_linear=cfg.kp_linear,
            ki_linear=cfg.ki_linear,
            kd_linear=cfg.kd_linear,
            kp_angular=cfg.kp_angular,
            ki_angular=cfg.ki_angular,
            kd_angular=cfg.kd_angular,
            wheel_base=cfg.wheel_base,
            wheel_radius=cfg.wheel_radius,
            num_envs=env.num_envs,
            device=env.device,
            output_limits=cfg.output_limits,
        )

        # Time step
        self._dt = env.physics_dt * env.cfg.decimation

        # Action scale (applied after PID)
        self._scale = torch.tensor(cfg.scale, device=env.device).unsqueeze(0)

        # Storage for processed actions
        self._raw_actions = torch.zeros(env.num_envs, 2, device=env.device)  # (vx, wz) commands
        self._processed_actions = torch.zeros(env.num_envs, 5, device=env.device)  # Full 5 DOF

    @property
    def action_dim(self) -> int:
        """Dimension of action space (vx, wz)."""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions (vx, wz) received from policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions (5 DOF) after PID control."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process velocity commands through PID controller.

        Args:
            actions: Velocity commands [vx, wz]. Shape: (num_envs, 2)
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Get current velocity from robot (use root linear/angular velocity)
        current_lin_vel = self._robot.data.root_lin_vel_b[:, 0]  # vx in base frame
        current_ang_vel = self._robot.data.root_ang_vel_b[:, 2]  # wz in base frame
        vel_current = torch.stack([current_lin_vel, current_ang_vel], dim=-1)

        # Compute wheel velocities using PID
        full_action = self._pid_controller.compute_full_action(
            vel_cmd=actions,
            vel_current=vel_current,
            dt=self._dt,
            arm_action=0.0,  # Keep arm stationary
            gripper_action=None,  # Keep grippers stationary
        )

        # Apply scaling
        self._processed_actions[:] = full_action * self._scale

    def apply_actions(self):
        """Apply processed actions to robot joints."""
        # Set joint velocity targets
        self._robot.set_joint_velocity_target(self._processed_actions)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset PID controller state.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        super().reset(env_ids)

        # Reset PID controller
        self._pid_controller.reset(env_ids)


@configclass
class VelocityPIDActionTermCfg(ActionTermCfg):
    """Configuration for PID-based velocity action term."""

    class_type: type = VelocityPIDActionTerm
    """Class implementing the action term."""

    # PID gains for linear velocity control
    kp_linear: float = 2.0
    """Proportional gain for linear velocity."""
    ki_linear: float = 0.1
    """Integral gain for linear velocity."""
    kd_linear: float = 0.05
    """Derivative gain for linear velocity."""

    # PID gains for angular velocity control
    kp_angular: float = 2.0
    """Proportional gain for angular velocity."""
    ki_angular: float = 0.1
    """Integral gain for angular velocity."""
    kd_angular: float = 0.05
    """Derivative gain for angular velocity."""

    # Robot kinematics
    wheel_base: float = 0.2
    """Distance between left and right wheels (meters)."""
    wheel_radius: float = 0.05
    """Radius of wheels (meters)."""

    # Output limits
    output_limits: tuple[float, float] | None = None
    """Wheel velocity limits (rad/s). None for no limits."""

    # Scaling
    scale: list[float] | float = [1.0, 1.0, 1.0, 1.0, 1.0]
    """Scaling factors for [left_wheel, right_wheel, arm, left_grip, right_grip]."""
