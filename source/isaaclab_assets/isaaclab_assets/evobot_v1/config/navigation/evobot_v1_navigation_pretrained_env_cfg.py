# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation environment using pre-trained balance policy for evobot_v1.

This configuration uses a pre-trained balance policy as low-level controller
and trains a high-level navigation policy on top of it.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..balance.evobot_v1_env_cfg_balance import EvobotV1EnvCfgBalance, ActionCfg as BalanceActionCfg
from ...mdp import (
    obs_body_pitch,
    obs_body_roll,
    obs_body_yaw,
    lin_vel_b,
    angl_vel_b,
    obs_pos_world,
    reset_when_fall,
)
from .mdp.pre_trained_policy_action import PreTrainedBalancePolicyActionCfg
from .mdp.rewards import (
    goal_progress_reward,
    velocity_towards_goal,
    heading_alignment_reward,
    lateral_velocity_penalty,
    yaw_rate_penalty,
    joint_velocity_penalty,
    position_reached_bonus,
    upright_reward,
    tilt_penalty,
)


# Load low-level balance environment config
LOW_LEVEL_ENV_CFG = EvobotV1EnvCfgBalance()


@configclass
class LowLevelObservationsCfg(ObsGroup):
    """Observations for the low-level balance policy.

    This must match the observation space that the balance policy was trained on.
    Based on evobot_v1_env_cfg_balance.py ObservationsCfg.PolicyCfg.

    The balance policy expects these observations in order:
    1. IMU data: lin_acc (3), ang_vel (3), orientation (4), projected_gravity (3)
    2. Body pose (7)
    3. Joint states: pos (5), vel (5), effort (5)
    4. Last action (5)
    Total: 40 dimensions
    """

    # IMU observations (matching balance policy)
    imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
    imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
    imu_orientation = ObsTerm(func=mdp.imu_orientation)
    imu_projected_gravity = ObsTerm(func=mdp.imu_projected_gravity)

    # Body pose
    body_pose_w = ObsTerm(func=mdp.body_pose_w)

    # Joint states
    joint_pos = ObsTerm(func=mdp.joint_pos)
    joint_vel = ObsTerm(func=mdp.joint_vel)
    joint_effort = ObsTerm(func=mdp.joint_effort)

    # Previous actions
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ActionsCfg:
    """Action configuration using pre-trained balance policy."""

    pre_trained_policy_action: PreTrainedBalancePolicyActionCfg = PreTrainedBalancePolicyActionCfg(
        asset_name="robot",
        # Path to pre-trained balance policy
        # Run play.py first to export: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        #   --task Isaac-Evobot-V1-Balance --num_envs 1 'agent.load_run=<run_name>' 'agent.load_checkpoint="model_.pt"'
        # Then update this path to point to the exported policy.pt
        policy_path="logs/rsl_rl/evobot_v1_balance/2026-01-09_08-48-26/exported/policy.pt",
        low_level_decimation=1,
        low_level_actions=BalanceActionCfg.wheel_effort,
        low_level_observations=LowLevelObservationsCfg(),
        velocity_scale=0.5,
        turn_scale=0.5,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for navigation policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """High-level observations for navigation."""

        # Robot state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Navigation command (target position)
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for navigation."""

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class CommandsCfg:
    """Command configuration for navigation."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(5.0, 5.0),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class RewardsCfg:
    # =====================================================
    # Termination
    # =====================================================
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-300.0,
    )

    # =====================================================
    # Core navigation (QUAN TRONG NHAT)
    # =====================================================
    goal_progress = RewTerm(
        func=goal_progress_reward,
        weight=5.0,
        params={"command_name": "pose_command"},
    )

    velocity_to_goal = RewTerm(
        func=velocity_towards_goal,
        weight=0.5,
        params={"command_name": "pose_command"},
    )

    # =====================================================
    # Orientation (nhe, chi ho tro)
    # =====================================================
    heading_alignment = RewTerm(
        func=heading_alignment_reward,
        weight=0.2,
        params={"command_name": "pose_command"},
    )

    # =====================================================
    # Stability penalties
    # =====================================================
    lateral_drift = RewTerm(
        func=lateral_velocity_penalty,
        weight=0.2,
    )

    yaw_rate = RewTerm(
        func=yaw_rate_penalty,
        weight=0.2,
    )

    joint_vel = RewTerm(
        func=joint_velocity_penalty,
        weight=0.001,  # Reduced weight since L2 penalty can be large
    )

    # =====================================================
    # Sparse success reward
    # =====================================================
    reached_bonus = RewTerm(
        func=position_reached_bonus,
        weight=10.0,
        params={
            "threshold": 0.3,
            "command_name": "pose_command",
        },
    )

    # ===============================
    # Balance (QUAN TRONG)
    # ===============================
    upright = RewTerm(
        func=upright_reward,
        weight=3.0,
    )

    tilt = RewTerm(
        func=tilt_penalty,
        weight=0.5,
    )


@configclass
class TerminationsCfg:
    """Termination configuration for navigation."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(
        func=reset_when_fall,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_tilt_angle": 1.0,  # ~57 degrees
        },
    )


@configclass
class EvobotV1NavigationPretrainedEnvCfg(ManagerBasedRLEnvCfg):
    """Navigation environment using pre-trained balance policy."""

    # Use the same scene as balance task
    scene = LOW_LEVEL_ENV_CFG.scene

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # Use same simulation settings as low-level env
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation

        # Higher decimation for navigation (low-level runs faster)
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 5

        # Episode length matches command resampling
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        # Viewer settings
        self.viewer.eye = (0.0, 8.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # Scene settings
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0


@configclass
class EvobotV1NavigationPretrainedEnvCfgPlay(EvobotV1NavigationPretrainedEnvCfg):
    """Play configuration for navigation with pre-trained policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
