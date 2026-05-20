"""Legged locomotion environment for Legged Robot V3.

Robot: 2-legged wheeled robot (5-bar parallel linkage per leg)
Task: Track velocity commands using leg-based locomotion.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Leg \\
        --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V3-Leg \\
        --num_envs 4
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.sensors import ImuCfg, ContactSensorCfg
from isaaclab.envs.mdp import actions, observations, events, rewards, terminations, commands
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_vel

from ..legged_v3_cfg import LEGGED_ROBOT_V3_CFG
from .. import mdp


# ─────────────────────────── Scene ────────────────────────────────────────────

@configclass
class LeggedV3SceneCfg(InteractiveSceneCfg):
    """Scene for legged_v3 velocity tracking."""

    num_envs: int = 1

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Rough terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    noise_range=(0.01, 0.05),
                    noise_step=0.01,
                    border_width=0.25,
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.2,
                    slope_range=(0.0, 0.2),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.02, 0.08),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: Articulation = LEGGED_ROBOT_V3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # IMU on base_link (URDF root after merge_fixed_joints).
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.01,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )

    # URDF with merge_fixed_joints=True → all links are flat under /Robot/.
    # Three separate sensors to avoid body-count mismatch across legs.

    # Sensor 1: base_link
    contact_forces_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        debug_vis=False,
    )

    # Sensor 2: right side (pad_link_right, thigh_right_1/2, calf_right_link_1/2, wheel_link_right)
    contact_forces_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*right.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

    # Sensor 3: left side (pad_link_left, thigh_left_1/2, calf_left_link_1/2, wheel_link_left)
    contact_forces_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*left.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """Joint effort control for 8 active DOF (pad + thigh_1 + calf_1 + wheel, both sides)."""

    # Leg joints: position control
    leg_pos = actions.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "pad_joint_right",
            "thigh_joint_right_1",
            "calf_joint_right_1",
            "pad_joint_left",
            "thigh_joint_left_1",
            "calf_joint_left_1",
        ],
        scale=1.0,
    )

    # Wheels: velocity control
    wheel_vel = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_joint_right", "wheel_joint_left"],
        scale=10.0,
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:
    """Velocity commands for differential drive + height command."""

    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.5,
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-5.0, 5.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )

    height_command = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(3.0, 6.0),
        make_quat_unique=False,
        debug_vis=False,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.4, 0.7),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Observations for policy and critic."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observations (onboard sensors only — deployable on real robot)."""

        imu_lin_acc           = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel           = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation       = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)

        joint_pos   = ObservationTermCfg(func=observations.joint_pos)
        joint_vel   = ObservationTermCfg(func=observations.joint_vel)

        last_action = ObservationTermCfg(func=observations.last_action)

        velocity_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
        )

        height_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "height_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Critic observations (privileged — full simulator state)."""

        root_pos_w     = ObservationTermCfg(func=observations.root_pos_w)
        root_quat_w    = ObservationTermCfg(func=observations.root_quat_w)
        root_lin_vel_w = ObservationTermCfg(func=observations.root_lin_vel_w)
        root_ang_vel_w = ObservationTermCfg(func=observations.root_ang_vel_w)
        base_lin_vel   = ObservationTermCfg(func=observations.base_lin_vel)

        imu_lin_acc           = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel           = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation       = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)

        joint_pos    = ObservationTermCfg(func=observations.joint_pos)
        joint_vel    = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)

        last_action = ObservationTermCfg(func=observations.last_action)

        velocity_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
        )

        height_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "height_command"},
        )

        current_time   = ObservationTermCfg(func=observations.current_time_s)
        remaining_time = ObservationTermCfg(func=observations.remaining_time_s)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ─────────────────────────── Events ───────────────────────────────────────────

@configclass
class EventCfg:
    """Reset events with domain randomization."""

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
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
            },
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────

# Active leg joints only (passive chain follows via closed loop)
_LEG_JOINTS = ["pad_joint_.*", "thigh_joint_.*_1", "calf_joint_.*_1"]

@configclass
class RewardCfg:
    """Reward terms."""

    termination_penalty = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-200.0,
    )

    track_base_height_exp = RewardTermCfg(
        func=mdp.rewards.track_base_height_exp,
        weight=1.0,
        params={
            "command_name": "height_command",
            "std": 0.05,
        },
    )

    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp_vel.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp_vel.track_ang_vel_z_world_exp,
        weight=5.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    upright = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=10.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
        },
    )

    lin_vel_z_l2 = RewardTermCfg(
        func=rewards.lin_vel_z_l2,
        weight=-0.5,
    )

    feet_air_time_left = RewardTermCfg(
        func=mdp_vel.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "velocity_command",
            "sensor_cfg": SceneEntityCfg("contact_forces_left", body_names=["wheel_link_left"]),
            "threshold": 0.4,
        },
    )

    feet_air_time_right = RewardTermCfg(
        func=mdp_vel.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "velocity_command",
            "sensor_cfg": SceneEntityCfg("contact_forces_right", body_names=["wheel_link_right"]),
            "threshold": 0.4,
        },
    )

    feet_slide_left = RewardTermCfg(
        func=mdp_vel.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_left", body_names=["wheel_link_left"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel_link_left"]),
        },
    )

    feet_slide_right = RewardTermCfg(
        func=mdp_vel.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_right", body_names=["wheel_link_right"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel_link_right"]),
        },
    )

    stand_still = RewardTermCfg(
        func=mdp_vel.stand_still_joint_deviation_l1,
        weight=-1.0,
        params={
            "command_name": "velocity_command",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS),
        },
    )

    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.01,
    )


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 2,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.05,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 100.0,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # Terminate when base body touches ground
    illegal_contact_base = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_base",
                body_names=["base_link"],
            ),
        },
    )

    # Terminate when any right leg link (excluding wheel) touches ground
    illegal_contact_right = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_right",
                body_names=["pad_link_right", "thigh_right_1", "calf_right_link_1",
                            "thigh_right_2", "calf_right_link_2"],
            ),
        },
    )

    # Terminate when any left leg link (excluding wheel) touches ground
    illegal_contact_left = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_left",
                body_names=["pad_link_left", "thigh_left_1", "calf_left_link_1",
                            "thigh_left_2", "calf_left_link_2"],
            ),
        },
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV3LegEnvCfg(ManagerBasedRLEnvCfg):
    """Environment config for legged_v3 velocity tracking."""

    scene:        LeggedV3SceneCfg = LeggedV3SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionCfg        = ActionCfg()
    commands:     CommandsCfg      = CommandsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardCfg        = RewardCfg()
    terminations: TerminationsCfg  = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2           # control @ 100 Hz (sim 200 Hz / 2)
        self.episode_length_s = 40.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
