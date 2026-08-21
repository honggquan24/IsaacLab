# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity balance environment configuration for Evobot V1.

This extends the balance task with command-following capabilities,
allowing the robot to navigate to random target positions while
maintaining balance.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import *  # noqa: F403
from isaaclab.envs.mdp import actions, commands, events, observations, rewards, terminations
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    ImuCfg,
)
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_v

from .. import mdp
from ..evobot_cfg import EVOBOT_CFG
from ..mdp import (
    binary_gripper_tracking,  # Binary gripper tracking reward
    joint_angle_command_l2,
)


@configclass
class EvobotSceneConfig(InteractiveSceneCfg):
    """Scene configuration for the Evobot V1 environment."""

    num_envs: int = 1

    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # ground
    cfg_ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Add robot
    robot: Articulation = EVOBOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Add IMU sensor - mounted on top_link (upper body)
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/evobot/evobot/top_link",
        update_period=0.02,  # 50Hz to match control frequency
        gravity_bias=(0.0, 0.0, 0.0),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/evobot/evobot/.*")


@configclass
class ActionCfg:
    """Action configuration for joint effort control (ALL CONTINUOUS).
    Available joints in USD (5 DOF total):
    - left_wheel_joint (Revolute) - Continuous effort control
    - right_wheel_joint (Revolute) - Continuous effort control
    - arm_joint (Revolute) - Continuous effort control
    - left_gripper_joint (Prismatic) - Continuous effort control (tracks binary commands)
    - right_gripper_joint (Prismatic) - Continuous effort control (tracks binary commands)

    NOTE: Actions are continuous, but commands are binary (0 or 1).
    Policy learns to output continuous values that track binary targets.
    """

    all_joints = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "left_wheel_joint",
            "right_wheel_joint",
            "arm_joint",
            "left_gripper_joint",
            "right_gripper_joint",
        ],
        scale={
            "left_wheel_joint": 400.0,  # Match effort_limit
            "right_wheel_joint": 400.0,
            "arm_joint": 200.0,
            "left_gripper_joint": 80.0,  # Matched with trained config
            "right_gripper_joint": 80.0,  # Matched with trained config
        },
    )


@configclass
class CommandsCfg:
    """Commands for gripper fine-tuning: slow velocity, frequent gripper changes."""

    # Velocity command - SLOWER and STABLE (not main focus)
    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),  # LONGER stable periods
        rel_standing_envs=0.7,
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

    # Arm command - KEEP STABLE (minimal changes)
    arm_ee_pose = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link",
        resampling_time_range=(5.0, 7.0),  # LONGER periods - arm stays stable
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-math.pi, math.pi),
        ),
    )

    # BINARY GRIPPER COMMANDS - z is binary (0 or 1), other dims continuous
    grip_ee_pose_left = mdp.BinaryGripperCommandCfg(
        asset_name="robot",
        body_name="gripper",
        resampling_time_range=(1.5, 3.0),  # FAST changes for practice
        debug_vis=False,
        prob_open=0.5,  # Equal probability of open (1.0) / close (0.0)
        ranges=mdp.BinaryGripperCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used
            pos_y=(0.0, 0.0),  # Not used
            # pos_z is BINARY (0.0 or 1.0) - sampled via prob_open
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    grip_ee_pose_right = mdp.BinaryGripperCommandCfg(
        asset_name="robot",
        body_name="gripper_01",
        resampling_time_range=(1.5, 3.0),
        debug_vis=False,
        prob_open=0.5,
        ranges=mdp.BinaryGripperCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation configuration for the policy."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group."""

        # observation terms (order preserved)
        base_lin_vel = ObservationTermCfg(func=mdp_v.base_lin_vel)  # noise=Unoise(n_min=-0.1, n_max=0.1)
        # base_ang_vel = ObservationTermCfg(func=mdp_v.base_ang_vel) # noise=Unoise(n_min=-0.1, n_max=0.1)
        # projected_gravity = ObservationTermCfg(
        #     func=mdp_v.projected_gravity,
        # )

        # IMU sensors
        imu_lin_acc = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation = ObservationTermCfg(func=observations.imu_orientation)

        # Joint states
        joint_pos = ObservationTermCfg(func=observations.joint_pos)
        joint_vel = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)

        # Previous actions (for smoothness)
        last_action = ObservationTermCfg(func=observations.last_action)

        # Commands
        velocity_command = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
        )

        arm_ee_pose_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "arm_ee_pose"},
        )

        grip_ee_pose_left_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "grip_ee_pose_left"},
        )

        grip_ee_pose_right_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "grip_ee_pose_right"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        # observation terms (order preserved)
        base_lin_vel = ObservationTermCfg(func=mdp_v.base_lin_vel)  # noise=Unoise(n_min=-0.1, n_max=0.1)
        # base_ang_vel = ObservationTermCfg(func=mdp_v.base_ang_vel) # noise=Unoise(n_min=-0.1, n_max=0.1)
        # projected_gravity = ObservationTermCfg(
        #     func=mdp_v.projected_gravity,
        # )

        # IMU sensors
        imu_lin_acc = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation = ObservationTermCfg(func=observations.imu_orientation)

        # Joint states
        joint_pos = ObservationTermCfg(func=observations.joint_pos)
        joint_vel = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)

        # Previous actions (for smoothness)
        last_action = ObservationTermCfg(func=observations.last_action)

        # Commands
        velocity_command = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
        )

        arm_ee_pose_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "arm_ee_pose"},
        )

        grip_ee_pose_left_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "grip_ee_pose_left"},
        )

        grip_ee_pose_right_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "grip_ee_pose_right"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Event configuration - Minimal randomization for gripper fine-tuning."""

    # Reset joints to default (minimal noise)
    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (-0.01, 0.01),  # Very small noise
            "velocity_range": (0.0, 0.0),  # No velocity noise
        },
    )

    # Reset base to default pose (minimal noise)
    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.12, 0.12),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "linear": (0.0, 0.0),
                "angular": (0.0, 0.0),
            },
        },
    )

    # external_push_arm = EventTermCfg(
    #     func=mdp_v.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(1.0, 2.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[
    #                 "arm_link",
    #                 "gripper_.*",
    #             ],
    #         ),
    #         "force_range": (-100.0, 100.0),
    #         "torque_range": (-100.0, 100.0),
    #     },
    # )


@configclass
class RewardCfg:
    """REWARD DESIGN: Balance + Velocity Tracking."""

    # (1) Survival
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=2.0,
    )

    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-100.0,
    )

    # (3) Command tracking - REDUCE velocity weights (keep stable, not focus)
    lin_vel_tracking = RewardTermCfg(
        func=rewards.track_lin_vel_xy_exp,
        weight=25.0,  # REDUCED from 20.0 - keep moving but not priority
        params={
            "command_name": "velocity_command",
            "std": 0.5,
        },
    )

    ang_vel_tracking = RewardTermCfg(
        func=rewards.track_ang_vel_z_exp,
        weight=25.0,  # REDUCED from 20.0
        params={
            "command_name": "velocity_command",
            "std": 0.5,
        },
    )

    # Joint angle tracking - Arm (keep stable, not focus)
    arm_joint_tracking = RewardTermCfg(
        func=joint_angle_command_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_joint"),
            "command_name": "arm_ee_pose",
        },
    )

    # BINARY GRIPPER TRACKING - MAIN FOCUS (Exponential reward for binary targets)
    grip_ee_tracking_left = RewardTermCfg(
        func=binary_gripper_tracking,
        weight=10.0,  # POSITIVE weight - exponential reward (higher is better)
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="left_gripper_joint"),
            "command_name": "grip_ee_pose_left",
            "joint_limits": (0.0, 0.2),  # Gripper joint range in meters
        },
    )

    grip_ee_tracking_right = RewardTermCfg(
        func=binary_gripper_tracking,
        weight=10.0,  # POSITIVE weight - exponential reward
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="right_gripper_joint"),
            "command_name": "grip_ee_pose_right",
            "joint_limits": (0.0, 0.2),
        },
    )

    # Smooth
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.005,
    )


@configclass
class TerminationsCfg:
    """Terminations: Lenient for gripper fine-tuning (focus on learning, not terminating)."""

    # TIME OUT (longer episodes for gripper practice)
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    # FALL DOWN (more lenient)
    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.25,  # Very low threshold - only stop if completely fallen
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # BAD ORIENTATION (very lenient - focus on gripper, not balance)
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 1.2,  # Almost horizontal before terminating
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # Contact illegal
    left_grip_contact = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={"threshold": 5000.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper.*")},
    )

    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 1000.0,  # rad/s - Giới hạn vận tốc góc tối đa
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


@configclass
class EvobotGripperFineTuneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for gripper fine-tuning (velocity + manipulation)."""

    # Scene
    scene: EvobotSceneConfig = EvobotSceneConfig(
        num_envs=1,
        env_spacing=2.0,
    )

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # General
        self.sim.device = "gpu"
        self.sim.use_fabric = True

        self.decimation = 1
        self.episode_length_s = 10.0  # Longer episodes for gripper practice
        # Physics
        self.sim.dt = 1 / 60.0

        # Viewer
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
