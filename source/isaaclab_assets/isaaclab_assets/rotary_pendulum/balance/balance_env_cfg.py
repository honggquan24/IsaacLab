# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the Rotary Pendulum V2 swing-up/balance task.

Task: Control the pivot motor (Revolute_1) to swing up and balance
the pendulum (Revolute_2) in the inverted (upright) position.

Observations (10D):
- sin(theta1), cos(theta1): Pivot angle (trigonometric)
- sin(theta2), cos(theta2): Pendulum angle (trigonometric)
- dtheta1, dtheta2: Joint velocities
- last_action (1D): Previous pivot torque
- generated_commands (3D): Target heading command (pos_x=0, pos_y=0, heading) - fixed base

Actions (1D):
- Torque applied to pivot motor (Revolute_1)
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.utils import configclass

from .. import mdp
from ..rotary_pendulum_cfg import ROTARY_PENDULUM_CFG


@configclass
class RotaryPendulumSceneConfig(InteractiveSceneCfg):
    """Scene configuration for the Rotary Pendulum V2 environment."""

    num_envs: int = 1

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    cfg_ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Rotary pendulum robot
    robot: Articulation = ROTARY_PENDULUM_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
    )


@configclass
class ActionCfg:
    """Action configuration - only pivot motor is actuated.

    Joints:
    - Revolute_1 (pivot motor): Actuated - torque control
    - Revolute_2 (pendulum): Passive - no actuation
    """

    # Pivot motor torque control
    pivot_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Revolute_1"],
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for the policy.

    Uses trigonometric representation of angles to avoid discontinuities.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group (10D total)."""

        # Trigonometric angle representation (avoids discontinuity at +-pi)
        joint_pos_sin = ObservationTermCfg(
            func=mdp.obs_joint_pos_sin,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_pos_cos = ObservationTermCfg(
            func=mdp.obs_joint_pos_cos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )

        # Joint velocities
        joint_vel = ObservationTermCfg(
            func=mdp.obs_joint_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )

        # Previous action for smoothness
        last_action = ObservationTermCfg(func=observations.last_action)

        # Commanded heading target (3D: pos_x=0, pos_y=0, heading) - fixed base
        generated_commands = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "pose_cmd"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Critic observation group (privileged info)."""

        # Raw joint positions (privileged)
        joint_pos = ObservationTermCfg(
            func=mdp.obs_joint_pos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )

        # Trigonometric angles
        joint_pos_sin = ObservationTermCfg(
            func=mdp.obs_joint_pos_sin,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_pos_cos = ObservationTermCfg(
            func=mdp.obs_joint_pos_cos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )

        # Joint velocities
        joint_vel = ObservationTermCfg(
            func=mdp.obs_joint_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )

        # Previous action
        last_action = ObservationTermCfg(func=observations.last_action)

        # Commanded heading target (3D: pos_x=0, pos_y=0, heading) - fixed base
        generated_commands = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "pose_cmd"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandCfg:
    """Command configuration for pivot rotation target around Z axis.

    Samples a target heading angle for the pivot joint (Revolute_1).
    The agent must balance the pendulum while tracking the target pivot angle.
    """

    pose_cmd = commands.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(5.0, 10.0),
        debug_vis=True,
        ranges=commands.UniformPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Fixed base - no position command
            pos_y=(0.0, 0.0),  # Fixed base - no position command
            heading=(-math.pi / 2, math.pi / 2),  # Target pivot angle range
        ),
    )


@configclass
class EventCfg:
    """Event configuration for environment resets."""

    # Reset joints with random offsets
    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardCfg:
    """Reward terms for pendulum swing-up and balance task."""

    # (1) Pendulum upright - main reward signal
    pendulum_upright = RewardTermCfg(
        func=mdp.pendulum_upright_reward,
        weight=10.0,
        params={
            "pendulum_joint_idx": 1,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # L2 penalty for deviation from upright position (pi radians)
    # pendulum_upright = RewardTermCfg(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-10.0,  # Negative weight = penalize squared error
    #     params={
    #         "target": math.pi,
    #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
    #     },
    # )

    # (2) Balance bonus - extra reward when upright AND stable
    balance_bonus = RewardTermCfg(
        func=mdp.balance_reward,
        weight=20.0,
        params={
            "vel_threshold": 0.5,
            "angle_range": 10.0,  # Scale from 170 deg to 180 deg
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
        },
    )

    # (3) Pendulum velocity penalty (conditional: penalize when upright, reward when swinging)
    # pendulum_vel_penalty = RewardTermCfg(
    #     func=mdp.pendulum_angular_velocity_penalty,
    #     weight=0.1,
    #     params={
    #         "swing_scale": 0.01,
    #         "balance_scale": -10.0,
    #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
    #     },
    # )

    pendulum_vel_penalty = RewardTermCfg(
        func=mdp.pendulum_angular_velocity_penalty,
        weight=-0.1,
        params={
            "swing_scale": 0.01,
            "balance_scale": 1.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
        },
    )

    # # (4) Pivot velocity penalty (conditional: penalize when upright, reward when swinging)
    # pivot_vel_penalty = RewardTermCfg(
    #     func=mdp.pivot_velocity_penalty,
    #     weight=0.1,
    #     params={
    #         "swing_scale": 0.01,
    #         "balance_scale": -10.0,
    #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_1"),
    #     },
    # )

    # (5) Pivot heading tracking - only active when balanced (upright + stable)
    pivot_heading_tracking = RewardTermCfg(
        func=mdp.pivot_heading_tracking_reward,
        weight=-12.0,
        params={
            "command_name": "pose_cmd",
            "vel_threshold": 0.5,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_1"),
        },
    )

    # (6) Energy penalty (conditional: penalize when upright, reward when swinging)
    # energy = RewardTermCfg(
    #     func=mdp.energy_penalty,
    #     weight=0.1,
    #     params={
    #         "swing_scale": 0.01,
    #         "balance_scale": -10.0,
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    # (7) Action smoothness penalty (conditional on pendulum state)
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2_pendulum,
        weight=0.1,
        params={
            "swing_scale": 0.01,
            "balance_scale": -1.0,
        },
    )

    # (8) Survival reward
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=1.0,
    )

    # (9) Termination penalty
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-1000.0,
    )


@configclass
class TerminationsCfg:
    """Termination configuration for Rotary Pendulum V2 environment."""

    # Timeout - episode ends after episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    # Pivot exceeds rotation limit (prevent endless spinning)
    pivot_limit = TerminationTermCfg(
        func=mdp.reset_when_pivot_exceeds_limit,
        params={
            "pivot_joint_idx": 0,
            "max_pivot_angle": math.pi / 2,  # 90 degrees (same as heading range ±90°)
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


@configclass
class RotaryPendulumBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Rotary Pendulum V2 balance environment."""

    # Scene settings
    scene: RotaryPendulumSceneConfig = RotaryPendulumSceneConfig(
        num_envs=1,
        env_spacing=1.0,
    )

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.sim.device = "gpu"
        self.sim.use_fabric = True

        self.decimation = 1  # Control freq = 60 Hz (dt=1/60, decimation=1)
        self.episode_length_s = 10  # 10 seconds per episode

        # Viewer settings
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # Physics timestep
        self.sim.dt = 1 / 60  # 60 Hz physics
        self.sim.render_interval = self.decimation
