"""Hierarchical navigation environment for Legged Robot V3.

High-level policy: outputs velocity commands (vx, vy, omega) → fed to pre-trained low-level policy.
Low-level policy:  controls joints directly to track velocity commands (train with Isaac-Legged-V3-Wheel).

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Navigation \\
        --num_envs 1024 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V3-Navigation \\
        --num_envs 4
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import actions, observations, events, rewards, terminations, commands
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..legged_v3_cfg import LEGGED_ROBOT_V3_CFG
from .pre_trained_policy_action import PreTrainedPolicyAction, PreTrainedPolicyActionCfg
from . import rewards as nav_rewards


# ─────────────────────────── Low-level observation group ──────────────────────
# Mirrors the wheel locomotion policy's observation space.
# PreTrainedPolicyAction will remap `actions` and `velocity_commands` at runtime.

@configclass
class LowLevelObsCfg(ObservationGroupCfg):
    """Observations fed into the pre-trained low-level locomotion policy."""

    imu_lin_acc           = ObservationTermCfg(func=observations.imu_lin_acc)
    imu_ang_vel           = ObservationTermCfg(func=observations.imu_ang_vel)
    imu_orientation       = ObservationTermCfg(func=observations.imu_orientation)
    imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)
    joint_pos             = ObservationTermCfg(func=observations.joint_pos)
    joint_vel             = ObservationTermCfg(func=observations.joint_vel)
    actions               = ObservationTermCfg(func=observations.last_action)   # remapped at runtime
    velocity_commands     = ObservationTermCfg(                                 # remapped at runtime
        func=observations.generated_commands,
        params={"command_name": "velocity_command"},
    )
    height_cmd            = ObservationTermCfg(
        func=observations.generated_commands,
        params={"command_name": "height_command"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


# ─────────────────────────── Scene ────────────────────────────────────────────

@configclass
class NavSceneCfg(InteractiveSceneCfg):

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.3,
        ),
        debug_vis=False,
    )

    robot: Articulation = LEGGED_ROBOT_V3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.01,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class NavActionCfg:
    """High-level policy outputs velocity commands → passed to low-level policy."""

    locomotion = PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=MISSING,          # set to trained model_*.pt path before training
        low_level_decimation=4,       # low-level runs at 50 Hz; high-level at 50/4 = 12.5 Hz
        low_level_actions=actions.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "pad_joint_right", "thigh_joint_right_1", "calf_joint_right_1",
                "pad_joint_left",  "thigh_joint_left_1",  "calf_joint_left_1",
            ],
            scale=5.0,
        ),
        low_level_observations=LowLevelObsCfg(),
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class NavCommandsCfg:
    """Goal position command (x, y in world frame + yaw)."""

    goal = commands.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 20.0),
        debug_vis=True,
        ranges=commands.UniformPose2dCommandCfg.Ranges(
            pos_x=(-5.0, 5.0),
            pos_y=(-5.0, 5.0),
            heading=(-3.14159, 3.14159),
        ),
    )

    # Needed by the low-level policy but won't be resampled (remapped at runtime).
    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),   # effectively never resample
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

    height_command = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(1e9, 1e9),
        make_quat_unique=False,
        debug_vis=False,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.5, 0.5),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class NavObsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """High-level policy observations: goal + robot state."""

        # Goal relative to robot in robot frame (dx, dy, distance, heading_error)
        goal_pos_b = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "goal"},
        )
        root_lin_vel_b = ObservationTermCfg(func=observations.base_lin_vel)
        root_ang_vel_w = ObservationTermCfg(func=observations.root_ang_vel_w)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)
        last_action = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────── Events ───────────────────────────────────────────

@configclass
class NavEventCfg:

    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.02, 0.02),
        },
    )

    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────

@configclass
class NavRewardCfg:

    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-200.0)

    # Move toward goal
    progress = RewardTermCfg(
        func=nav_rewards.progress_toward_goal,
        weight=5.0,
        params={
            "command_name": "goal",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Exponential bonus near goal
    goal_proximity = RewardTermCfg(
        func=nav_rewards.goal_distance_exp,
        weight=2.0,
        params={
            "command_name": "goal",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 2.0,
        },
    )

    # Bonus on arrival
    goal_reached = RewardTermCfg(
        func=nav_rewards.goal_reached,
        weight=50.0,
        params={
            "command_name": "goal",
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.3,
        },
    )

    # Face toward goal
    heading = RewardTermCfg(
        func=nav_rewards.heading_toward_goal,
        weight=1.0,
        params={
            "command_name": "goal",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Smoothness penalty
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.05)
    lin_vel_z_l2 = RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-0.5)


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class NavTermCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg(name="robot")},
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV3WheelNavigationEnvCfg(ManagerBasedRLEnvCfg):

    scene:        NavSceneCfg    = NavSceneCfg(num_envs=1, env_spacing=8.0)
    observations: NavObsCfg      = NavObsCfg()
    actions:      NavActionCfg   = NavActionCfg()
    commands:     NavCommandsCfg = NavCommandsCfg()
    events:       NavEventCfg    = NavEventCfg()
    rewards:      NavRewardCfg   = NavRewardCfg()
    terminations: NavTermCfg     = NavTermCfg()

    def __post_init__(self):
        self.decimation = 20          # high-level @ ~10 Hz (sim 200 Hz / 20)
        self.episode_length_s = 30.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
