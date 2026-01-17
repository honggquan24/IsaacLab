# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking environment using pre-trained balance policy for evobot_v1.

This configuration uses a pre-trained balance+velocity policy as low-level controller
and trains a high-level velocity command policy on top of it.

The key difference from velocity_env_cfg.py:
- velocity_env_cfg.py: Learns both balance AND velocity tracking (low-level)
- hierarchical_vel_env_cfg.py: Uses pre-trained low-level policy, only learns velocity command generation (high-level)
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import RewardTermCfg
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
    velocity_heading_alignment,
    PreTrainedBalancePolicyActionCfg,
)

# Import velocity env config from correct location
from .velocity_env_cfg import EvobotV1VelocityBalanceEnvCfg, ActionCfg
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
        policy_path="logs/rsl_rl/evobot_v1_velocity/2026-01-15_01-19-39/exported/policy.pt",
        low_level_decimation=1,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.all_joints,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        # CRITICAL: These scales determine additional velocity modulation on top of low-level policy
        #
        # HOW IT WORKS:
        # 1. High-level outputs velocity commands [vx, vy, omega] in range [-1, 1]
        # 2. These are passed to low-level policy via base_velocity_cmd observation
        # 3. Low-level policy outputs joint actions based on velocity command + balance
        # 4. THEN we optionally ADD velocity modulation: vx*scale ± omega*scale
        #
        # RECOMMENDED VALUES:
        # - If low-level tracks velocity well: velocity_scale=0.0, turn_scale=0.0
        # - For fine-tuning/boosting: velocity_scale=0.1-0.5, turn_scale=0.1-0.5
        # - If low-level doesn't track velocity: velocity_scale=3.0-5.0, turn_scale=2.0-3.0
        velocity_scale=0.01,  # No additional modulation (trust low-level policy)
        turn_scale=0.01,      # No additional modulation (trust low-level policy)
    )


@configclass
class ObservationsCfg:
    """Observation configuration for velocity tracking policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """High-level observations for velocity command generation."""

        # Robot velocity (current state)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)

        # Orientation (để biết hướng mặt)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        # Velocity command (target velocity to track)
        velocity_command = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations (with privileged info)."""

        # Robot velocity
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)

        # Orientation
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        # Velocity command
        velocity_command = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
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
    """Command configuration for velocity control."""

    # Velocity commands với consideration cho balance
    # IMPORTANT: Match với low-level training distribution để tránh distribution mismatch!
    base_velocity = commands.UniformVelocity2DCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),  # TĂNG thời gian command (smooth hơn)
        # QUAN TRỌNG: Tùy chỉnh cho balance
        rel_standing_envs=0.2,     # 20% thời gian đứng yên (nhiều hơn để ổn định)

        heading_command=False,     # Dùng angular velocity (not heading)
        # debug_vis=True,

        # RANGE GIỐNG LOW-LEVEL TRAINING để tránh out-of-distribution
        ranges=commands.UniformVelocity2DCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),      # GIẢM range để trong comfort zone của low-level
            lin_vel_y=(0.0, 0.0),        # BỎ y-velocity (differential drive không đi ngang)
            ang_vel_z=(-1.0, 1.0),       # GIẢM angular velocity range
            heading=(0.0, 0.0),          # Không dùng heading mode
        ),
    )


@configclass
class RewardsCfg:
    """REWARD DESIGN: Velocity tracking (low-level lo balance, high-level focus velocity)"""

    # (1) MAIN REWARD: Track linear velocity command
    lin_vel_tracking = RewardTermCfg(
        func=rewards.track_lin_vel_xy_exp,
        weight=20.0,  # MAIN REWARD: Track vx command
        params={
            "command_name": "base_velocity",
            "std": 0.5,  # Exponential kernel std
        },
    )

    # (2) Track angular velocity command (rotation)
    ang_vel_tracking = RewardTermCfg(
        func=rewards.track_ang_vel_z_exp,
        weight=15.0,  # Track wz command
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # (3) Heading alignment: Robot xoay đúng hướng trước khi đi
    heading_alignment = RewardTermCfg(
        func=velocity_heading_alignment,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # (4) Penalty: Smooth actions
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.05,
    )

    # (5) Survival
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=1.0,
    )

    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-500.0,
    )

    # (6) Penalty: Undesired contacts
    undesired_contacts = RewardTermCfg(
        func=rewards.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="head_link|gripper.*"),
            "threshold": 10.0
        },
    )

    # (7) BONUS: Upright orientation (khuyến khích high-level giữ balance!)
    # Điều này giúp high-level tránh output commands làm robot ngã
    upright_bonus = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=5.0,  # Reward khi robot đứng thẳng (penalize tilt)
        params={"asset_cfg": SceneEntityCfg("robot")},
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
    
    # # 3. BAD ORIENTATION (khi nghiêng quá nhiều)
    # bad_orientation = TerminationTermCfg(
    #     func=terminations.bad_orientation,
    #     params={
    #         "limit_angle": math.pi / 1.2,  
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    
    
    # # Contact illegal 
    # arm_contact = TerminationTermCfg( 
    #     func=terminations.illegal_contact, 
    #     params={ 
    #         "threshold": 10.0, 
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="head_link") 
    #     }
    # ) 
    
    # left_grip_contact = TerminationTermCfg( 
    #     func=terminations.illegal_contact, 
    #     params={ 
    #         "threshold": 10.0, 
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper") 
    #     }
    # ) 
    
    # right_grip_contact = TerminationTermCfg( 
    #     func=terminations.illegal_contact, 
    #     params={ 
    #         "threshold": 10.0, 
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper_01") 
    #     }
    # ) 
    
    # joint_vel_limit = TerminationTermCfg(
    #     func=terminations.joint_vel_out_of_manual_limit,
    #     params={
    #         "max_velocity": 1000.0,  # rad/s - Giới hạn vận tốc góc tối đa
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    


@configclass
class EvobotV1VelocityPretrainedEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity tracking environment using pre-trained balance policy.

    High-level policy learns to track velocity commands.
    Low-level policy (pre-trained) handles balance and joint control.
    """

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

        # Higher decimation for high-level policy (low-level runs faster)
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 2

        # Episode length matches command resampling
        self.episode_length_s = 10

        # Viewer settings
        self.viewer.eye = (0.0, 8.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # Scene settings
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0


@configclass
class EvobotV1VelocityPretrainedEnvCfgPlay(EvobotV1VelocityPretrainedEnvCfg):
    """Play configuration for velocity tracking with pre-trained policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
