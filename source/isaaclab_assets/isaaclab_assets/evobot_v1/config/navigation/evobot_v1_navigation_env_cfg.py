"""Navigation environment configuration for Evobot V1.

This extends the balance task with command-following capabilities,
allowing the robot to navigate to random target positions while
maintaining balance.
"""

import math
from isaaclab.managers import ObservationTermCfg, RewardTermCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

# Import balance config as base
from ..env.evobot_v1_env_cfg_balance import (
    EvobotV1EnvCfgBalance,
    EvobotV1SceneConfig,
    ObservationsCfg,
    RewardCfg,
)

# Import navigation MDP functions
from . import mdp as nav_mdp


@configclass
class CommandsCfg:
    """Command configuration for navigation.

    Generates random target positions for the robot to navigate to.
    The pose_command includes [x, y, heading] target.
    """

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # Fixed 10s resampling for training
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-2.0, 2.0),  # Target x position range (meters)
            pos_y=(-2.0, 2.0),  # Target y position range (meters)
            heading=(-math.pi, math.pi),  # Target heading range (radians)
        ),
    )


@configclass
class NavigationObservationsCfg(ObservationsCfg):
    """Navigation observations extend balance observations.

    Adds pose command observations so the policy can track targets.
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Policy observation group with command."""

        # Inherit all balance observations (IMU, joints, actions, etc.)
        # Add command observation
        pose_command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "pose_command"})


@configclass
class NavigationRewardCfg(RewardCfg):
    """Navigation rewards extend balance rewards.

    Keeps all balance rewards and adds navigation-specific terms.
    """

    # Balance rewards inherited from parent:
    # - alive (2.0)
    # - terminating (-100.0)
    # - rpy_alignment (10.0)
    # - action_rate (-1.0)
    # - joint_vel (-0.5)
    # - ang_vel_xy (-0.5)
    # - base_height (-1.0)

    # NEW: Navigation-specific rewards

    # (1) Position tracking - coarse
    position_tracking = RewardTermCfg(
        func=nav_mdp.rewards.position_command_error_tanh,
        weight=3.0,
        params={
            "std": 1.5,  # Loose tolerance for exploration
            "command_name": "pose_command",
        },
    )

    # (2) Position tracking - fine
    position_tracking_fine = RewardTermCfg(
        func=nav_mdp.rewards.position_command_error_tanh,
        weight=2.0,
        params={
            "std": 0.3,  # Tight tolerance for precision
            "command_name": "pose_command",
        },
    )

    # (3) Heading tracking penalty
    heading_tracking = RewardTermCfg(
        func=nav_mdp.rewards.heading_command_error_abs,
        weight=-0.3,  # Negative weight (function returns positive error)
        params={
            "command_name": "pose_command",
        },
    )

    # (4) Goal reached bonus
    position_reached = RewardTermCfg(
        func=nav_mdp.rewards.position_reached_bonus,
        weight=5.0,  # Significant bonus for reaching target
        params={
            "threshold": 0.3,  # Within 30cm
            "command_name": "pose_command",
        },
    )


@configclass
class EvobotV1NavigationEnvCfg(EvobotV1EnvCfgBalance):
    """Navigation environment configuration.

    Extends the balance environment with command generation and
    navigation reward terms.
    """

    # Use base scene (no changes needed - inherits robot, sensors, etc.)
    scene: EvobotV1SceneConfig = EvobotV1SceneConfig()

    # Add command manager
    commands: CommandsCfg = CommandsCfg()

    # Use navigation observations (includes pose command)
    observations: NavigationObservationsCfg = NavigationObservationsCfg()

    # Use navigation rewards (balance + navigation terms)
    rewards: NavigationRewardCfg = NavigationRewardCfg()

    def __post_init__(self):
        """Post initialization - extend episode length for navigation."""
        super().__post_init__()
        # Longer episodes for exploration and reaching targets
        self.episode_length_s = 30.0  # 30s vs 20s for balance


@configclass
class EvobotV1NavigationEnvCfgPlay(EvobotV1NavigationEnvCfg):
    """Navigation environment configuration for play/evaluation.

    Uses fixed command resampling for consistent evaluation.
    """

    def __post_init__(self):
        super().__post_init__()
        # Fixed command resampling for evaluation
        self.commands.pose_command.resampling_time_range = (2.0, 2.0)
