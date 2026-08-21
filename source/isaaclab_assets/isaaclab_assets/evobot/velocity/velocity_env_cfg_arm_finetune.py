# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity balance environment configuration for Evobot V1.

This extends the balance task with command-following capabilities,
allowing the robot to navigate to random target positions while
maintaining balance.
"""

# ruff: noqa: F405  (env cfg dùng tên đến từ `import *` của package mdp trong cùng dự án)

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
        update_period=1 / 60,
        gravity_bias=(0.0, 0.0, 0.0),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/evobot/evobot/.*")


@configclass
class ActionCfg:
    """Action configuration for joint effort control.
    Available joints in USD (5 DOF total):
    - left_wheel_joint (Revolute)
    - right_wheel_joint (Revolute)
    - arm_joint (Revolute)
    - left_gripper_joint (Prismatic)
    - right_gripper_joint (Prismatic)
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
            "arm_joint": 400.0,
            "left_gripper_joint": 0.0,
            "right_gripper_joint": 0.0,
        },
    )


@configclass
class CommandsCfg:
    """Velocity commands với consideration cho balance."""

    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),  # Giữ command 3-5s
        # QUAN TRỌNG: Tùy chỉnh cho balance
        rel_standing_envs=0.7,  # 30% thời gian đứng yên (tập balance tại chỗ)
        heading_command=False,  # FALSE = Dùng angular velocity (not heading angle)
        debug_vis=True,
        # RANGE AN TOÀN CHO BALANCE + Xoay
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),  # TỐC ĐỘ CHẬM để giữ balance
            lin_vel_y=(0.0, 0.0),  # BỎ y-velocity (differential drive không đi ngang)
            ang_vel_z=(-0.5, 0.5),  # Angular velocity range
            heading=(0.0, 0.0),  # Ignored khi heading_command=False
        ),
    )

    # Arm joint angle command (use yaw component as joint angle target)
    arm_ee_pose = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link",
        resampling_time_range=(3.0, 5.0),  # Change target every 3-5 seconds
        # debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used
            pos_y=(0.0, 0.0),  # Not used
            pos_z=(0.0, 0.0),  # Not used
            roll=(0.0, 0.0),  # Not used
            pitch=(0.0, 0.0),  # Not used
            yaw=(-0 / 2, 0 / 2),  # Use yaw as joint angle target (±90 degrees)
        ),
    )

    # Binary command - z is binary (0 or 1), other dims continuous
    grip_ee_pose_left = mdp.BinaryGripperCommandCfg(
        asset_name="robot",
        body_name="gripper",
        resampling_time_range=(1.5, 3.0),
        debug_vis=False,
        prob_open=0.5,  # Equal probability of open (1.0) / close (0.0)
        ranges=mdp.BinaryGripperCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            # pos_z is BINARY (0.0 or 1.0)
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
    """Event configuration for environment resets."""

    # Reset joints with small random offsets (avoid local minima)
    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
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

    # randomize_com = EventTermCfg(
    #     func=mdp_v.randomize_rigid_body_com,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[
    #                 "arm_link",
    #             ],
    #         ),
    #         "com_range": {
    #             "x": (-0.01, 0.01),
    #             "y": (-0.01, 0.01),
    #             "z": (-0.01, 0.01),
    #         },
    #     },
    # )

    # external_push_arm = EventTermCfg(
    #     func=mdp_v.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(3.0, 5.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[
    #                 "arm_link",
    #             ],
    #         ),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-10.0, 10.0),
    #     },
    # )


@configclass
class RewardCfg:
    """REWARD DESIGN: Balance + Velocity Tracking."""

    # (1) Survival
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=10.0,
    )

    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-5000.0,
    )

    # (3) Command tracking
    lin_vel_tracking = RewardTermCfg(
        func=mdp.track_lin_vel_xy_l2,
        weight=-10.0,
        params={
            "command_name": "velocity_command",
        },
    )

    ang_vel_tracking = RewardTermCfg(
        func=mdp.track_ang_vel_z_l2,
        weight=-10.0,
        params={
            "command_name": "velocity_command",
        },
    )

    # Joint angle tracking - Arm tracks commanded angle (from yaw component)
    arm_joint_tracking = RewardTermCfg(
        func=joint_angle_command_l2,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_joint"),
            "command_name": "arm_ee_pose",
        },
    )

    velocity_arm = RewardTermCfg(
        func=joint_vel_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_joint"),
        },
    )

    # Smooth
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.001,
    )


@configclass
class TerminationsCfg:
    # """Terminations: Strict cho balance, lenient cho velocity."""
    # 1. TIME OUT (normal)
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    # 2. FALL DOWN
    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.25,  # 8cm (thấp hơn chút để cho recovery chance)
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


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurriculumTermCfg(func=mdp_v.terrain_levels_vel)


@configclass
class EvobotArmFineTuneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration cho velocity control với balance constraints."""

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
        self.episode_length_s = 10.0
        # Physics
        self.sim.dt = 1 / 60.0

        # Viewer
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
