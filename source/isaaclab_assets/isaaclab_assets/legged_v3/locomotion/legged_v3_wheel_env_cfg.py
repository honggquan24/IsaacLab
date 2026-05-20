"""Wheeled locomotion environment for Legged Robot V3.

Robot: 2-legged wheeled robot (5-bar parallel linkage per leg)
Task: Track velocity commands using wheel-based locomotion while maintaining balance.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Wheel \\
        --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V3-Wheel \\
        --num_envs 4
"""

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


import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_vel

from ..legged_v3_cfg import LEGGED_ROBOT_V3_CFG
from .. import mdp

import math 

# ─────────────────────────── Scene ────────────────────────────────────────────

@configclass
class LeggedV3SceneCfg(InteractiveSceneCfg):
    """Scene for legged_v3 wheel velocity tracking."""

    num_envs: int = 1

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

    # IMU on base_link (URDF root after URDF conversion).
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.01,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )

    # URDF with merge_fixed_joints → all links flat under /Robot/.
    # Three separate sensors to avoid body-count mismatch.

    # contact_forces_base = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #     update_period=0.0,
    #     debug_vis=False,
    # )

    # contact_forces_right = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*right.*",
    #     update_period=0.0,
    #     history_length=3,
    #     track_air_time=True,
    #     debug_vis=False,
    # )

    # contact_forces_left = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*left.*",
    #     update_period=0.0,
    #     history_length=3,
    #     track_air_time=True,
    #     debug_vis=False,
    # )


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """Position control for leg joints + velocity control for wheels."""

    # Leg joints: position control (policy outputs target angles in rad)
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
        scale=5.0,
    )

    # Wheels: velocity control (policy outputs target angular velocity in rad/s)
    wheel_vel = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_joint_right", "wheel_joint_left"],
        scale=5.0,
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:
    """Velocity + height commands."""

    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.3,
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
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
            pos_z=(0.4, 0.6),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):
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
        last_action  = ObservationTermCfg(func=observations.last_action)
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
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
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

_LEG_JOINTS = ["pad_joint_.*", "thigh_joint_.*_1", "calf_joint_.*_1"]

@configclass
class RewardCfg:

    # ── Primary task ──────────────────────────────────────────────────────────
    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-500.0)

    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp_vel.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp_vel.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    track_base_height_exp = RewardTermCfg(
        func=mdp.rewards.track_base_height_exp,
        weight=2.0,
        params={"command_name": "height_command", "std": 0.05},
    )

    # ── Stability ─────────────────────────────────────────────────────────────
    upright = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=-5.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
        },
    )

    lin_vel_z_l2 = RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-1.0)

    # ── Joint / action smoothness ─────────────────────────────────────────────
    dof_pos_limits = RewardTermCfg(
        func=rewards.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )

    joint_deviation_pad = RewardTermCfg(
        func=rewards.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="pad_joint_.*")},
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

    joint_acc = RewardTermCfg(
        func=rewards.joint_acc_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )

    joint_torques = RewardTermCfg(
        func=rewards.joint_torques_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )

    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.01)


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 2,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # base_height = TerminationTermCfg(
    #     func=terminations.root_height_below_minimum,
    #     params={
    #         "minimum_height": 0.05,
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 120.0,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # illegal_contact_base = TerminationTermCfg(
    #     func=terminations.illegal_contact,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg(name="contact_forces_base", body_names=["base_link"]),
    #     },
    # )

    # illegal_contact_right = TerminationTermCfg(
    #     func=terminations.illegal_contact,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg(
    #             name="contact_forces_right",
    #             body_names=["pad_link_right", "hip_frame_link_right",
    #                         "thigh_right_1", "calf_right_link_1",
    #                         "thigh_right_2", "calf_right_link_2"],
    #         ),
    #     },
    # )

    # illegal_contact_left = TerminationTermCfg(
    #     func=terminations.illegal_contact,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg(
    #             name="contact_forces_left",
    #             body_names=["pad_link_left", "hip_frame_link_left",
    #                         "thigh_left_1", "calf_left_link_1",
    #                         "thigh_left_2", "calf_left_link_2"],
    #         ),
    #     },
    # )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV3WheelEnvCfg(ManagerBasedRLEnvCfg):

    scene:        LeggedV3SceneCfg = LeggedV3SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionCfg        = ActionCfg()
    commands:     CommandsCfg      = CommandsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardCfg        = RewardCfg()
    terminations: TerminationsCfg  = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4           # control @ 50 Hz (sim 200 Hz / 4)
        self.episode_length_s = 60.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
