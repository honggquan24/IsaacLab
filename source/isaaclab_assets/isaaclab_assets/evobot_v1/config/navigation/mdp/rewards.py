"""Navigation-specific reward functions for Evobot V1."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking position command using tanh.

    This reward uses tanh function to provide smooth, bounded rewards
    for position tracking. The std parameter controls how quickly the
    reward decays with distance error.

    Args:
        env: The RL environment.
        std: Standard deviation for tanh scaling (larger = more forgiving).
        command_name: Name of the command to track.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        Tanh-based position tracking reward for each environment.
    """
    # Get target position from command
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :2]  # Extract [x, y] position

    # Get current robot position
    robot = env.scene[asset_cfg.name]
    current_pos = robot.data.root_pos_w[:, :2]

    # Compute L2 distance error
    error = torch.norm(target_pos - current_pos, dim=1)

    # Tanh-based reward (smooth, bounded in [-1, 1])
    reward = torch.tanh(-error / std)
    return reward


def heading_command_error_abs(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for heading error (absolute value).

    Computes the absolute heading error between current robot heading
    and target heading from command. Returns positive error values
    (to be weighted negatively in reward config).

    Args:
        env: The RL environment.
        command_name: Name of the command to track.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        Absolute heading error for each environment.
    """
    # Get target heading from command
    command = env.command_manager.get_command(command_name)
    target_heading = command[:, 2]  # Extract yaw angle

    # Get current heading from robot orientation
    robot = env.scene[asset_cfg.name]
    current_quat = robot.data.root_quat_w

    # Convert quaternion to euler angles
    from isaaclab.utils.math import euler_xyz_from_quat
    _, _, current_yaw = euler_xyz_from_quat(current_quat)

    # Compute heading error (handle wrapping with wrap_to_pi)
    error = torch.abs(wrap_to_pi(target_heading - current_yaw))
    return error


def position_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Discrete bonus when robot reaches target position.

    Provides a bonus reward (1.0) when the robot is within the specified
    threshold distance from the target position.

    Args:
        env: The RL environment.
        threshold: Distance threshold for considering target reached (meters).
        command_name: Name of the command to track.
        asset_cfg: Scene entity configuration for the robot.

    Returns:
        Bonus value (1.0 if reached, 0.0 otherwise) for each environment.
    """
    # Get target and current position
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :2]

    robot = env.scene[asset_cfg.name]
    current_pos = robot.data.root_pos_w[:, :2]

    # Check if within threshold
    distance = torch.norm(target_pos - current_pos, dim=1)
    reached = distance < threshold

    return reached.float()


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi] range.

    Utility function to handle angle wrapping, ensuring that angles
    are always in the range [-π, π].

    Args:
        angle: Input angle tensor (radians).

    Returns:
        Wrapped angle in [-π, π] range.
    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi
