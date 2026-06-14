"""Hierarchical navigation environment cho Legged Robot V5.

High-level policy: xuất velocity command (vx, vy, omega) → đẩy vào policy
locomotion V5 đã train (chạy bên trong PreTrainedPolicyAction).
Low-level policy: bám velocity command (train bằng Isaac-Legged-V5-Wheel).

⚠️ TRƯỚC KHI TRAIN: phải set `policy_path` tới checkpoint locomotion đã export
(logs/rsl_rl/legged_v5_wheel_mimic/<run>/exported/policy.pt).

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V5-Navigation --num_envs 1024 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V5-Navigation --num_envs 4
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import observations, events, rewards, terminations, commands
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..legged_v5_cfg import LEGGED_V5_CFG
from .. import mdp
from ..locomotion.legged_v5_wheel_env_cfg import ActionCfg as LowLevelActionCfg
from .pre_trained_policy_action import PreTrainedPolicyActionCfg
from . import rewards as nav_rewards


# ───────────────────── Low-level observation (PHẢI mirror PolicyCfg locomotion V5) ──
# Thứ tự + nội dung phải khớp đúng obs lúc train Isaac-Legged-V5-Wheel.
# PreTrainedPolicyAction remap `last_action` và `velocity_cmd` lúc runtime.

@configclass
class LowLevelObsCfg(ObservationGroupCfg):
    base_lin_vel      = ObservationTermCfg(func=observations.base_lin_vel)
    projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
    joint_pos         = ObservationTermCfg(func=observations.joint_pos)
    joint_vel         = ObservationTermCfg(func=observations.joint_vel)
    last_action       = ObservationTermCfg(func=observations.last_action)          # remap runtime
    velocity_cmd      = ObservationTermCfg(                                          # remap runtime
        func=observations.generated_commands, params={"command_name": "velocity_command"},
    )
    height_cmd        = ObservationTermCfg(
        func=observations.generated_commands, params={"command_name": "height_command"},
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

    robot: Articulation = LEGGED_V5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class NavActionCfg:
    """High-level xuất velocity command → đẩy vào policy locomotion đã train."""

    locomotion = PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=MISSING,            # ⚠️ set tới policy.pt locomotion đã export
        low_level_decimation=2,         # khớp decimation=2 lúc train locomotion (100 Hz)
        low_level_actions=LowLevelActionCfg(),   # leg_pos (mimic) + wheel_vel của V5
        low_level_observations=LowLevelObsCfg(),
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class NavCommandsCfg:
    """Goal position (x, y world + heading) + height cố định cho low-level."""

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

    # velocity_command bị remap hoàn toàn (PreTrainedPolicyAction), không resample.
    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0),
        ),
    )

    # Height giữ cố định = chiều cao đứng (khớp lúc train low-level).
    height_command = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="base",
        resampling_time_range=(1e9, 1e9),
        make_quat_unique=False,
        debug_vis=False,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.35, 0.35),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


# ─────────────────────────── Observations (high-level) ────────────────────────

@configclass
class NavObsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        goal_pos_b        = ObservationTermCfg(
            func=observations.generated_commands, params={"command_name": "goal"},
        )
        root_lin_vel_b    = ObservationTermCfg(func=observations.base_lin_vel)
        root_ang_vel_w    = ObservationTermCfg(func=observations.root_ang_vel_w)
        projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
        last_action       = ObservationTermCfg(func=observations.last_action)

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

    progress = RewardTermCfg(
        func=nav_rewards.progress_toward_goal,
        weight=5.0,
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
    )
    goal_proximity = RewardTermCfg(
        func=nav_rewards.goal_distance_exp,
        weight=2.0,
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot"), "std": 2.0},
    )
    goal_reached = RewardTermCfg(
        func=nav_rewards.goal_reached,
        weight=50.0,
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot"), "threshold": 0.3},
    )
    heading = RewardTermCfg(
        func=nav_rewards.heading_toward_goal,
        weight=1.0,
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate  = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.05)
    lin_vel_z_l2 = RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-0.5)


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class NavTermCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=mdp.terminations.bad_orientation_from_default,
        params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg(name="robot")},
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV5NavigationEnvCfg(ManagerBasedRLEnvCfg):

    scene:        NavSceneCfg    = NavSceneCfg(num_envs=1, env_spacing=8.0)
    observations: NavObsCfg      = NavObsCfg()
    actions:      NavActionCfg   = NavActionCfg()
    commands:     NavCommandsCfg = NavCommandsCfg()
    events:       NavEventCfg    = NavEventCfg()
    rewards:      NavRewardCfg   = NavRewardCfg()
    terminations: NavTermCfg     = NavTermCfg()

    def __post_init__(self):
        self.decimation = 20          # high-level @ 10 Hz (sim 200 / 20)
        self.episode_length_s = 30.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
