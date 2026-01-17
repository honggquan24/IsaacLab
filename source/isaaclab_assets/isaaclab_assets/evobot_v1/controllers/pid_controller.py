# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PID Controller implementation for Evobot V1 velocity control."""

import torch
from typing import Optional


class PIDController:
    """Vectorized PID controller for batched environments.

    This implements a discrete-time PID controller with anti-windup and derivative filtering.
    All operations are vectorized to support multiple parallel environments.

    Args:
        kp: Proportional gain (scalar or tensor)
        ki: Integral gain (scalar or tensor)
        kd: Derivative gain (scalar or tensor)
        num_envs: Number of parallel environments
        device: Device to run computations on
        output_limits: Tuple of (min, max) for output saturation. None for no limits.
        integral_limits: Tuple of (min, max) for integral term saturation. None for no limits.
        derivative_filter: Low-pass filter coefficient for derivative term (0-1). 0 = no filtering.
    """

    def __init__(
        self,
        kp: float | torch.Tensor,
        ki: float | torch.Tensor,
        kd: float | torch.Tensor,
        num_envs: int,
        device: str,
        output_limits: Optional[tuple[float, float]] = None,
        integral_limits: Optional[tuple[float, float]] = None,
        derivative_filter: float = 0.0,
    ):
        self.device = device
        self.num_envs = num_envs

        # Convert gains to tensors
        self.kp = self._to_tensor(kp)
        self.ki = self._to_tensor(ki)
        self.kd = self._to_tensor(kd)

        # Output saturation limits
        self.output_limits = output_limits
        self.integral_limits = integral_limits

        # Derivative filter coefficient (0 = no filter, 1 = full filter)
        self.derivative_filter = derivative_filter

        # State variables (num_envs x action_dim)
        self.integral = None
        self.prev_error = None
        self.filtered_derivative = None

    def _to_tensor(self, value: float | torch.Tensor) -> torch.Tensor:
        """Convert scalar or tensor to device tensor."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.tensor(value, device=self.device)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset PID state for specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.integral is None:
            # First reset - initialize all states
            action_dim = 1 if self.kp.dim() == 0 else len(self.kp)
            self.integral = torch.zeros((self.num_envs, action_dim), device=self.device)
            self.prev_error = torch.zeros((self.num_envs, action_dim), device=self.device)
            self.filtered_derivative = torch.zeros((self.num_envs, action_dim), device=self.device)
        else:
            # Reset specific environments
            self.integral[env_ids] = 0.0
            self.prev_error[env_ids] = 0.0
            self.filtered_derivative[env_ids] = 0.0

    def compute(
        self,
        error: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute PID control output.

        Args:
            error: Current error (setpoint - measurement). Shape: (num_envs, action_dim) or (num_envs,)
            dt: Time step in seconds

        Returns:
            Control output. Shape: (num_envs, action_dim) or (num_envs,)
        """
        # Ensure error is 2D
        if error.dim() == 1:
            error = error.unsqueeze(-1)

        # Initialize if first call
        if self.integral is None:
            self.reset()

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        if self.integral_limits is not None:
            self.integral = torch.clamp(self.integral, self.integral_limits[0], self.integral_limits[1])
        i_term = self.ki * self.integral

        # Derivative term with optional filtering
        derivative = (error - self.prev_error) / dt
        if self.derivative_filter > 0:
            self.filtered_derivative = (
                self.derivative_filter * self.filtered_derivative +
                (1 - self.derivative_filter) * derivative
            )
            derivative = self.filtered_derivative
        d_term = self.kd * derivative

        # Combined output
        output = p_term + i_term + d_term

        # Output saturation
        if self.output_limits is not None:
            output = torch.clamp(output, self.output_limits[0], self.output_limits[1])

        # Store for next iteration
        self.prev_error = error.clone()

        # Return same shape as input
        if output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output


class VelocityPIDController:
    """PID controller for differential drive robot velocity control.

    Controls linear velocity (vx) and angular velocity (wz) using two independent PIDs.
    Converts velocity commands to left/right wheel velocities using differential drive kinematics.

    Args:
        kp_linear: Proportional gain for linear velocity
        ki_linear: Integral gain for linear velocity
        kd_linear: Derivative gain for linear velocity
        kp_angular: Proportional gain for angular velocity
        ki_angular: Integral gain for angular velocity
        kd_angular: Derivative gain for angular velocity
        wheel_base: Distance between left and right wheels (meters)
        wheel_radius: Radius of wheels (meters)
        num_envs: Number of parallel environments
        device: Device to run computations on
        output_limits: Wheel velocity limits (rad/s)
    """

    def __init__(
        self,
        kp_linear: float = 1.0,
        ki_linear: float = 0.1,
        kd_linear: float = 0.05,
        kp_angular: float = 1.0,
        ki_angular: float = 0.1,
        kd_angular: float = 0.05,
        wheel_base: float = 0.2,  # 20cm default
        wheel_radius: float = 0.05,  # 5cm default
        num_envs: int = 1,
        device: str = "cuda",
        output_limits: Optional[tuple[float, float]] = None,
    ):
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.num_envs = num_envs
        self.device = device

        # Create two independent PID controllers
        self.linear_pid = PIDController(
            kp=kp_linear,
            ki=ki_linear,
            kd=kd_linear,
            num_envs=num_envs,
            device=device,
            output_limits=output_limits,
            derivative_filter=0.1,  # Light filtering for smoother control
        )

        self.angular_pid = PIDController(
            kp=kp_angular,
            ki=ki_angular,
            kd=kd_angular,
            num_envs=num_envs,
            device=device,
            output_limits=output_limits,
            derivative_filter=0.1,
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset PID states for specified environments."""
        self.linear_pid.reset(env_ids)
        self.angular_pid.reset(env_ids)

    def compute(
        self,
        vel_cmd: torch.Tensor,
        vel_current: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute wheel velocities from velocity commands.

        Args:
            vel_cmd: Desired velocity [vx, wz]. Shape: (num_envs, 2)
            vel_current: Current velocity [vx, wz]. Shape: (num_envs, 2)
            dt: Time step in seconds

        Returns:
            Wheel velocities [left, right]. Shape: (num_envs, 2)
        """
        # Ensure inputs are 2D
        if vel_cmd.dim() == 1:
            vel_cmd = vel_cmd.unsqueeze(0)
        if vel_current.dim() == 1:
            vel_current = vel_current.unsqueeze(0)

        # Extract linear (vx) and angular (wz) velocities
        vx_cmd = vel_cmd[:, 0]
        wz_cmd = vel_cmd[:, 1]
        vx_current = vel_current[:, 0]
        wz_current = vel_current[:, 1]

        # Compute errors
        linear_error = vx_cmd - vx_current
        angular_error = wz_cmd - wz_current

        # PID control for linear and angular velocities
        vx_control = self.linear_pid.compute(linear_error, dt)
        wz_control = self.angular_pid.compute(angular_error, dt)

        # Convert to wheel velocities using differential drive kinematics
        # v_left = (vx - wz * wheel_base/2) / wheel_radius
        # v_right = (vx + wz * wheel_base/2) / wheel_radius
        v_left = (vx_control - wz_control * self.wheel_base / 2) / self.wheel_radius
        v_right = (vx_control + wz_control * self.wheel_base / 2) / self.wheel_radius

        # Stack to (num_envs, 2)
        wheel_velocities = torch.stack([v_left, v_right], dim=-1)

        return wheel_velocities

    def compute_full_action(
        self,
        vel_cmd: torch.Tensor,
        vel_current: torch.Tensor,
        dt: float,
        arm_action: Optional[torch.Tensor] = None,
        gripper_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute full action vector for evobot (5 DOF).

        Args:
            vel_cmd: Desired velocity [vx, wz]. Shape: (num_envs, 2)
            vel_current: Current velocity [vx, wz]. Shape: (num_envs, 2)
            dt: Time step
            arm_action: Arm joint action (default: 0). Shape: (num_envs,) or scalar
            gripper_action: Gripper actions (default: [0, 0]). Shape: (num_envs, 2)

        Returns:
            Full action [left_wheel, right_wheel, arm, left_grip, right_grip]. Shape: (num_envs, 5)
        """
        # Compute wheel velocities
        wheel_vel = self.compute(vel_cmd, vel_current, dt)

        # Handle arm action
        if arm_action is None:
            arm_action = torch.zeros(self.num_envs, device=self.device)
        elif isinstance(arm_action, (int, float)):
            arm_action = torch.full((self.num_envs,), arm_action, device=self.device)
        elif arm_action.dim() == 0:
            arm_action = arm_action.unsqueeze(0).expand(self.num_envs)

        # Handle gripper action
        if gripper_action is None:
            gripper_action = torch.zeros((self.num_envs, 2), device=self.device)
        elif gripper_action.dim() == 1:
            gripper_action = gripper_action.unsqueeze(0).expand(self.num_envs, -1)

        # Concatenate full action [left_wheel, right_wheel, arm, left_grip, right_grip]
        full_action = torch.cat([
            wheel_vel,  # (num_envs, 2)
            arm_action.unsqueeze(-1),  # (num_envs, 1)
            gripper_action,  # (num_envs, 2)
        ], dim=-1)

        return full_action
