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
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations, commands

# Import consolidated MDP components from root mdp folder
from ...mdp import (
    obs_body_pitch,
    obs_body_roll,
    obs_body_yaw,
    lin_vel_b,
    angl_vel_b,
    obs_pos_world,
    reset_when_fall,
    position_command_error_tanh,
    heading_command_error_abs,
    position_reached_bonus,
    goal_progress_reward,
    velocity_towards_goal,
    heading_alignment_reward,
    lateral_velocity_penalty,
    yaw_rate_penalty,
    joint_velocity_penalty,
    upright_reward,
    tilt_penalty,
    PreTrainedBalancePolicyActionCfg,
)

# Import velocity env config from correct location
from ..velocity.velocity_env_cfg import EvobotV1VelocityBalanceEnvCfg
from ..velocity.velocity_env_cfg import ActionCfg as VelocityBalanceActionCfg

# Import velocity-based locomotion MDP terms
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_v


# Load low-level balance environment config
LOW_LEVEL_ENV_CFG = EvobotV1VelocityBalanceEnvCfg()


@configclass
class ActionsCfg:
    """Action configuration using pre-trained balance policy."""

    pre_trained_policy_action: PreTrainedBalancePolicyActionCfg = PreTrainedBalancePolicyActionCfg(
        asset_name="robot",
        # Path to pre-trained balance policy
        # Run play.py first to export: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        #   --task Isaac-Evobot-V1-Balance --num_envs 1 'agent.load_run=<run_name>' 'agent.load_checkpoint="model_.pt"'
        # Then update this path to point to the exported policy.pt
        policy_path="logs/rsl_rl/evobot_v1_velocity/2026-01-14_11-14-00/exported/policy.pt",
        low_level_decimation=1,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.all_joints,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        velocity_scale=0.1,
        turn_scale=0.1,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for navigation policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """High-level observations for navigation."""

        # Robot state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        # Navigation command (target position)
        pose_command = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):
        """High-level observations for navigation."""

        # Robot state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        # Navigation command (target position)
        pose_command = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Event configuration for navigation."""

    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )
    
    # Reset base with small noise (increase robustness)
    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.12, 0.12),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "linear": (-0.05, 0.05),
                "angular": (-0.05, 0.05),
            },
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
        weight=5,
        params={"command_name": "pose_command"},
    )

    # =====================================================
    # Orientation (nhe, chi ho tro)
    # =====================================================
    heading_alignment = RewTerm(
        func=heading_alignment_reward,
        weight=2.5,
        params={"command_name": "pose_command"},
    )

    # =====================================================
    # Stability penalties
    # =====================================================
    lateral_drift = RewTerm(
        func=lateral_velocity_penalty,
        weight=2,
    )

    yaw_rate = RewTerm(
        func=yaw_rate_penalty,
        weight=2,
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
        weight=50.0,
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
    """Terminations: Strict cho balance, lenient cho velocity."""
    # 1. TIME OUT (normal)
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )
    
    # 2. FALL DOWN 
    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.15,  # 8cm (thấp hơn chút để cho recovery chance)
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # 3. BAD ORIENTATION (khi nghiêng quá nhiều)
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 1.2,  
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    
    # Contact illegal 
    arm_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 10.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="head_link") 
        }
    ) 
    
    left_grip_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 10.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper") 
        }
    ) 
    
    right_grip_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 10.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper_01") 
        }
    ) 
    
    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 1000.0,  # rad/s - Giới hạn vận tốc góc tối đa
            "asset_cfg": SceneEntityCfg(name="robot"),
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
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 4

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
