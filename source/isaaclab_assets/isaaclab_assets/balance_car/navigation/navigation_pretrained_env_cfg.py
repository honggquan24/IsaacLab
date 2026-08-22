# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation environment using pre-trained balance policy for the balance car.

This configuration uses a pre-trained balance policy as low-level controller
and trains a high-level navigation policy on top of it.
"""

# ruff: noqa: F405  (env cfg dùng tên đến từ `import *` của package mdp trong cùng dự án)

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

from ..balance_env_cfg import BalanceCarEnvCfg
from ..mdp.observations import angl_vel_b, lin_vel_b, obs_body_pitch, obs_body_roll, obs_body_yaw
from ..mdp.terminations import reset_when_fall
from .mdp.pre_trained_policy_action import PreTrainedBalancePolicyActionCfg, latest_exported_policy
from .mdp.rewards import *  # noqa: F403

# Load low-level balance environment config
LOW_LEVEL_ENV_CFG = BalanceCarEnvCfg()


@configclass
class LowLevelObservationsCfg(ObsGroup):
    """Observations for the low-level balance policy.

    This must match the observation space that the balance policy was trained on.
    Based on balance_env_cfg.py ObservationsCfg.PolicyCfg (without obs_pos_w).
    """

    # observation terms (order preserved)
    joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    pitch_angl_p = ObsTerm(
        func=obs_body_pitch,
        params={"asset_cfg": SceneEntityCfg("imu")},
    )
    pitch_angl_r = ObsTerm(
        func=obs_body_roll,
        params={"asset_cfg": SceneEntityCfg("imu")},
    )
    pitch_angl_y = ObsTerm(
        func=obs_body_yaw,
        params={"asset_cfg": SceneEntityCfg("imu")},
    )
    l_vel = ObsTerm(
        func=lin_vel_b,
        params={"asset_cfg": SceneEntityCfg("imu")},
    )
    a_vel = ObsTerm(
        func=angl_vel_b,
        params={"asset_cfg": SceneEntityCfg("imu")},
    )
    # Chỗ giữ sẵn cho lệnh vận tốc; PreTrainedBalancePolicyAction ghi đè func lúc khởi tạo để
    # nó trả về action của tầng cao. Thứ tự term phải khớp đúng PolicyCfg của tầng thấp.
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ActionsCfg:
    """Action configuration using pre-trained balance policy."""

    pre_trained_policy_action: PreTrainedBalancePolicyActionCfg = PreTrainedBalancePolicyActionCfg(
        asset_name="robot",
        # Tự lấy run mới nhất của tầng thấp. Train Isaac-Balance-Car rồi chạy play.py một lần
        # để nó export ra logs/rsl_rl/carbalance_ppo/<run>/exported/policy.pt là dùng được ngay,
        # không phải quay lại sửa file này.
        policy_path=latest_exported_policy("carbalance_ppo"),
        low_level_decimation=1,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_effort,
        low_level_observations=LowLevelObservationsCfg(),
        # debug_vis=True,
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

    reset_pole_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_[1-2]"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CommandsCfg:
    """Command configuration for navigation."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(5.0, 5.0),
        # debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            heading=(-math.pi, math.pi),  # sẽ bị IGNORE
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
    # Core navigation (QUAN TRỌNG NHẤT)
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
    # Orientation (nhẹ, chỉ hỗ trợ)
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
        weight=0.1,
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
    # Balance (QUAN TRỌNG)
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
    fall = DoneTerm(func=reset_when_fall)


@configclass
class BalanceCarNavigationPretrainedEnvCfg(ManagerBasedRLEnvCfg):
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
class BalanceCarNavigationPretrainedEnvCfg_PLAY(BalanceCarNavigationPretrainedEnvCfg):
    """Play configuration for navigation with pre-trained policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
