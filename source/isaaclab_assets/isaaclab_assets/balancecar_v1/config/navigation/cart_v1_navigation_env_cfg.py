# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation environment configuration for balance car v1."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from ...cart_v1_cfg import CART_BALANCE_CFG
from ...mdp.observations import (
    angl_vel_b,
    lin_vel_b,
    obs_body_pitch,
    obs_body_roll,
    obs_body_yaw,
    obs_pos_world,
)
from ...mdp.rewards import (
    bonus_reward,
    reward_angle_r,
    reward_angle_y,
    reward_li_vel,
    reward_roll_rate,
    reward_vel,
)
from ...mdp.terminations import reset_when_fall
from ..navigation.mdp.rewards import (
    position_command_error_tanh,
    heading_command_error_abs,
    position_reached_bonus,
)


@configclass
class CartV1NavigationSceneCfg(InteractiveSceneCfg):
    """Scene configuration for balance car navigation."""

    num_envs: int = 1

    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Robot
    robot = CART_BALANCE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # IMU sensor
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/balance_robot/balance_robot/balance_body",
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.2),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        update_period=0.0,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action configuration for navigation."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Revolute_1", "Revolute_2"],
        scale={
            "Revolute_1": 100.0,
            "Revolute_2": 100.0,
        },
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for navigation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        # Robot state observations
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # IMU observations
        pitch_angle = ObsTerm(
            func=obs_body_pitch,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        roll_angle = ObsTerm(
            func=obs_body_roll,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        yaw_angle = ObsTerm(
            func=obs_body_yaw,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        linear_vel = ObsTerm(
            func=lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        angular_vel = ObsTerm(
            func=angl_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )

        # Position observation
        robot_pos = ObsTerm(
            func=obs_pos_world,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Navigation command (target position)
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        pitch_angle = ObsTerm(
            func=obs_body_pitch,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        roll_angle = ObsTerm(
            func=obs_body_roll,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        yaw_angle = ObsTerm(
            func=obs_body_yaw,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        linear_vel = ObsTerm(
            func=lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        angular_vel = ObsTerm(
            func=angl_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        robot_pos = ObsTerm(
            func=obs_pos_world,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        pose_command = ObsTerm(
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

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.1, 0.1)},
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     },
    # )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_[1-2]"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.01, 0.01),
        },
    )


@configclass
class CommandsCfg:
    """Command configuration for navigation."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(2.0, 2.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-2.0, 2.0),
            pos_y=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class RewardsCfg:
    """Reward configuration for navigation."""

    # Balance rewards (keep upright)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # Balance angle rewards
    balance_roll = RewTerm(func=reward_angle_r, weight=2.0)
    balance_yaw = RewTerm(func=reward_angle_y, weight=0.2)
    balance_roll_rate = RewTerm(func=reward_roll_rate, weight=0.3)

    # Navigation rewards
    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=3.0,
        params={"std": 1.5, "command_name": "pose_command"},
    )
    position_tracking_fine = RewTerm(
        func=position_command_error_tanh,
        weight=2.0,
        params={"std": 0.3, "command_name": "pose_command"},
    )
    heading_tracking = RewTerm(
        func=heading_command_error_abs,
        weight=-0.3,
        params={"command_name": "pose_command"},
    )
    position_reached = RewTerm(
        func=position_reached_bonus,
        weight=5.0,
        params={"threshold": 0.3, "command_name": "pose_command"},
    )

    # Velocity penalties
    # joint_vel_penalty = RewTerm(func=reward_vel, weight=0.1)
    # linear_vel_penalty = RewTerm(func=reward_li_vel, weight=0.1)


@configclass
class TerminationsCfg:
    """Termination configuration for navigation."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=reset_when_fall)


@configclass
class CartV1NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for balance car navigation environment."""

    scene: CartV1NavigationSceneCfg = CartV1NavigationSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 5.0

        self.viewer.eye = (0.0, 8.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation


@configclass
class CartV1NavigationEnvCfg_PLAY(CartV1NavigationEnvCfg):
    """Play configuration for balance car navigation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
