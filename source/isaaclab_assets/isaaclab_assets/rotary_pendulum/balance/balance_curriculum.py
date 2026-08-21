# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum Learning Configuration for Rotary Pendulum V2 Swing-Up + Balance Task.

CURRICULUM STAGES:
==================

Stage 1: SWING-UP ONLY (learn to get pendulum upright)
-------------------------------------------------------
- Simpler reward structure (no heading tracking)
- Longer episodes (20s) for exploration
- Larger pivot range (±180°) for swing-up
- Random initial conditions
- Focus: Get pendulum from hanging (0) to upright (π)

Stage 2: BALANCE + HEADING TRACKING (full task)
------------------------------------------------
- Add heading tracking reward
- Tighter termination (±90°)
- Load checkpoint from Stage 1
- Focus: Maintain balance while tracking pivot heading commands

TRAINING WORKFLOW:
==================

1. Train Stage 1:
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task=Isaac-Rotary-Pendulum-Balance-Stage1 \
       --num_envs 4096 \
       --headless

2. Train Stage 2 (load Stage 1 checkpoint):
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
       --task=Isaac-Rotary-Pendulum-Balance-Stage2 \
       --num_envs 4096 \
       --resume --load_run=<stage1_run_name> \
       --checkpoint=model_<best>.pt \
       --headless
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
    """Action configuration - only pivot motor is actuated."""

    pivot_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Revolute_1"],
        scale=1.0,
    )


# ============================================================================
# STAGE 1: SWING-UP ONLY
# ============================================================================


@configclass
class Stage1ObservationsCfg:
    """Stage 1 observations - NO heading command (simpler)."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group (7D total) - simplified for swing-up."""

        # Trigonometric angle representation
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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Critic observation group (privileged info)."""

        joint_pos = ObservationTermCfg(
            func=mdp.obs_joint_pos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_pos_sin = ObservationTermCfg(
            func=mdp.obs_joint_pos_sin,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_pos_cos = ObservationTermCfg(
            func=mdp.obs_joint_pos_cos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.obs_joint_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        last_action = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Stage1EventCfg:
    """Stage 1 events - WITH randomization for better exploration."""

    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (-0.2, 0.2),  # Random initial angle ±0.2 rad
            "velocity_range": (-0.5, 0.5),  # Random initial velocity
        },
    )


@configclass
class Stage1RewardCfg:
    """Stage 1 rewards - SWING-UP ONLY (no heading tracking)."""

    # (1) Pendulum upright - cos-based reward (better for swing-up)
    pendulum_upright = RewardTermCfg(
        func=mdp.pendulum_upright_reward,
        weight=5.0,  # Strong positive signal
        params={
            "pendulum_joint_idx": 1,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # (2) Balance bonus - extra reward when stable AND upright
    balance_bonus = RewardTermCfg(
        func=mdp.balance_reward,
        weight=10.0,  # Lower than stage 2 (swing-up is priority)
        params={
            "vel_threshold": 0.5,
            "angle_range": 20.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
        },
    )

    # (3) Pendulum velocity (encourage during swing, penalize when upright)
    pendulum_vel = RewardTermCfg(
        func=mdp.pendulum_angular_velocity_penalty,
        weight=0.05,  # Lower weight for exploration
        params={
            "swing_scale": 0.5,  # Small reward for velocity during swing
            "balance_scale": -2.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
        },
    )

    # (4) Energy penalty (encourage during swing, penalize when upright)
    energy = RewardTermCfg(
        func=mdp.energy_penalty,
        weight=0.05,
        params={
            "swing_scale": 0.5,
            "balance_scale": -2.0,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # (5) Survival reward
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=1.0,
    )

    # (6) Termination penalty
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-500.0,  # Lower penalty for exploration
    )


@configclass
class Stage1TerminationsCfg:
    """Stage 1 terminations - MORE LENIENT for swing-up exploration."""

    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    # Larger pivot range for swing-up
    pivot_limit = TerminationTermCfg(
        func=mdp.reset_when_pivot_exceeds_limit,
        params={
            "pivot_joint_idx": 0,
            "max_pivot_angle": math.pi,  # ±180° - full range for swing-up
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


@configclass
class RotaryPendulumBalanceStage1EnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1: Swing-up only configuration."""

    scene: RotaryPendulumSceneConfig = RotaryPendulumSceneConfig(
        num_envs=1,
        env_spacing=1.0,
    )

    observations: Stage1ObservationsCfg = Stage1ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    events: Stage1EventCfg = Stage1EventCfg()
    rewards: Stage1RewardCfg = Stage1RewardCfg()
    terminations: Stage1TerminationsCfg = Stage1TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.sim.device = "gpu"
        self.sim.use_fabric = True

        self.decimation = 1  # 60 Hz control
        self.episode_length_s = 20  # Longer episodes for exploration

        # Viewer settings
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # Physics timestep
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation


# ============================================================================
# STAGE 2: BALANCE + HEADING TRACKING
# ============================================================================


@configclass
class Stage2ObservationsCfg:
    """Stage 2 observations - WITH heading command."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group (10D total)."""

        joint_pos_sin = ObservationTermCfg(
            func=mdp.obs_joint_pos_sin,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_pos_cos = ObservationTermCfg(
            func=mdp.obs_joint_pos_cos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.obs_joint_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        last_action = ObservationTermCfg(func=observations.last_action)

        # NOW include heading command
        generated_commands = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "pose_cmd"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Critic observation group."""

        joint_pos = ObservationTermCfg(
            func=mdp.obs_joint_pos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_pos_sin = ObservationTermCfg(
            func=mdp.obs_joint_pos_sin,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_pos_cos = ObservationTermCfg(
            func=mdp.obs_joint_pos_cos,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.obs_joint_vel,
            params={"asset_cfg": SceneEntityCfg(name="robot")},
        )
        last_action = ObservationTermCfg(func=observations.last_action)
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
class Stage2CommandCfg:
    """Stage 2 commands - heading tracking."""

    pose_cmd = commands.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(5.0, 10.0),
        debug_vis=True,
        ranges=commands.UniformPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            heading=(-math.pi / 2, math.pi / 2),  # ±90° heading range
        ),
    )


@configclass
class Stage2RewardCfg:
    """Stage 2 rewards - FULL TASK (swing-up + heading tracking)."""

    # (1) Pendulum upright
    pendulum_upright = RewardTermCfg(
        func=mdp.pendulum_upright_reward,
        weight=5.0,
        params={
            "pendulum_joint_idx": 1,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # (2) Balance bonus - higher weight now
    balance_bonus = RewardTermCfg(
        func=mdp.balance_reward,
        weight=20.0,  # Higher emphasis on staying balanced
        params={
            "vel_threshold": 0.5,
            "angle_range": 20.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
        },
    )

    # (3) Pendulum velocity penalty
    pendulum_vel_penalty = RewardTermCfg(
        func=mdp.pendulum_angular_velocity_penalty,
        weight=0.1,
        params={
            "swing_scale": 1.0,
            "balance_scale": -2.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_2"),
        },
    )

    # (4) Pivot velocity penalty
    pivot_vel_penalty = RewardTermCfg(
        func=mdp.pivot_velocity_penalty,
        weight=0.1,
        params={
            "swing_scale": 1.0,
            "balance_scale": -2.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_1"),
        },
    )

    # (5) NEW: Pivot heading tracking - only active when balanced
    pivot_heading_tracking = RewardTermCfg(
        func=mdp.pivot_heading_tracking_reward,
        weight=-2.0,
        params={
            "command_name": "pose_cmd",
            "vel_threshold": 0.5,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="Revolute_1"),
        },
    )

    # (6) Energy penalty
    energy = RewardTermCfg(
        func=mdp.energy_penalty,
        weight=0.1,
        params={
            "swing_scale": 1.0,
            "balance_scale": -2.0,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # (7) Action smoothness
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2_pendulum,
        weight=0.1,
        params={
            "swing_scale": 1.0,
            "balance_scale": -2.0,
        },
    )

    # (8) Survival
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=1.0,
    )

    # (9) Termination penalty
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-1000.0,  # Higher penalty for failure
    )


@configclass
class Stage2TerminationsCfg:
    """Stage 2 terminations - TIGHTER for heading tracking."""

    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    # Tighter pivot range for controlled tracking
    pivot_limit = TerminationTermCfg(
        func=mdp.reset_when_pivot_exceeds_limit,
        params={
            "pivot_joint_idx": 0,
            "max_pivot_angle": math.pi / 2,  # ±90° - match heading range
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


@configclass
class RotaryPendulumBalanceStage2EnvCfg(ManagerBasedRLEnvCfg):
    """Stage 2: Full task with heading tracking."""

    scene: RotaryPendulumSceneConfig = RotaryPendulumSceneConfig(
        num_envs=1,
        env_spacing=1.0,
    )

    observations: Stage2ObservationsCfg = Stage2ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    commands: Stage2CommandCfg = Stage2CommandCfg()
    events: Stage1EventCfg = Stage1EventCfg()  # Keep randomization
    rewards: Stage2RewardCfg = Stage2RewardCfg()
    terminations: Stage2TerminationsCfg = Stage2TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.sim.device = "gpu"
        self.sim.use_fabric = True

        self.decimation = 1
        self.episode_length_s = 15  # Slightly shorter for tracking task

        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
