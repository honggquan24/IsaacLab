"""Single-run auto-curriculum environment for Legged Robot V3.

Robot: 2-legged wheeled (robot_legged_v3)

One training run, joints unlocked progressively:
  Phase 0  (iter    0 –  999): Wheels only (knee/thigh/hip frozen)
  Phase 1  (iter 1000 – 1999): + Knee unlocked
  Phase 2  (iter 2000 – 2999): + Thigh unlocked
  Phase 3  (iter 3000+):       + Hip unlocked (full 8-DOF)

Key design:
  - All 8 joints in action space from step 0 → fixed NN input/output size.
  - Frozen joints: ImplicitActuatorCfg(stiffness=5000) + JointPositionActionCfg(scale=0).
  - CurriculumCfg calls unlock_joint_phases() which:
      (a) writes_joint_stiffness_to_sim(20) to restore normal stiffness,
      (b) sets action term _scale = 1.0 to enable control.
  - All reward terms active from start; frozen joints naturally give near-zero penalty.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Legged-V3-Curriculum \
        --num_envs 4096 --headless
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg,
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

from .legged_v3_curr_cfg import LEGGED_CURRICULUM_ROBOT_CFG
from .. import mdp as legged_mdp
from .mdp import unlock_joint_phases


# ──────────────────────────────────────────────────────────────────────────────
# Scene
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene for curriculum training."""

    num_envs: int = 1

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: Articulation = LEGGED_CURRICULUM_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/base",
        update_period=0.01,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )

    contact_forces_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/base",
        update_period=0.0,
        debug_vis=False,
    )

    contact_forces_right_leg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/right_leg/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

    contact_forces_left_leg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/left_leg/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Actions — fixed NN output size throughout training
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumActionCfg:
    """8-DOF action space. Frozen joints start with scale=0; enabled by curriculum.

    Action breakdown (matches full-control case from day 1):
      wheel_vel  : 2 outputs → always active (velocity control)
      knee_pos   : 2 outputs → scale=0 until iter 1000, then scale=1
      thigh_pos  : 2 outputs → scale=0 until iter 2000, then scale=1
      hip_pos    : 2 outputs → scale=0 until iter 3000, then scale=1
    """

    # Wheels: always active
    wheel_vel = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale={"left_wheel_joint": 50.0, "right_wheel_joint": 50.0},
    )

    # Knee: frozen until phase 1 (scale=0 → target=init_pos every step)
    knee_pos = actions.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_knee_joint", "right_knee_joint"],
        scale=0.0,
    )

    # Thigh: frozen until phase 2
    thigh_pos = actions.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_thigh_joint", "right_thigh_joint"],
        scale=0.0,
    )

    # Hip: frozen until phase 3
    hip_pos = actions.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_joint", "right_hip_joint"],
        scale=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumCommandsCfg:
    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.02,
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
        body_name="base",
        resampling_time_range=(3.0, 6.0),
        make_quat_unique=False,
        debug_vis=False,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.4, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Observations — fixed size throughout training
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumObsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        imu_lin_acc           = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel           = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation       = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)
        joint_pos             = ObservationTermCfg(func=observations.joint_pos)
        joint_vel             = ObservationTermCfg(func=observations.joint_vel)
        last_action           = ObservationTermCfg(func=observations.last_action)
        velocity_cmd          = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
        )
        height_cmd            = ObservationTermCfg(
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
        imu_lin_acc    = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel    = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation       = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)
        joint_pos      = ObservationTermCfg(func=observations.joint_pos)
        joint_vel      = ObservationTermCfg(func=observations.joint_vel)
        joint_effort   = ObservationTermCfg(func=observations.joint_effort)
        last_action    = ObservationTermCfg(func=observations.last_action)
        velocity_cmd   = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
        )
        height_cmd     = ObservationTermCfg(
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


# ──────────────────────────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumEventCfg:

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


# ──────────────────────────────────────────────────────────────────────────────
# Rewards — all terms present from start; frozen joints naturally give ~0 penalty
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumRewardCfg:

    termination_penalty = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-200.0,
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
        func=legged_mdp.rewards.rpy_alignment_imu,
        weight=-10.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
        },
    )

    lin_vel_z_l2 = RewardTermCfg(
        func=rewards.lin_vel_z_l2,
        weight=-0.5,
    )

    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.01,
    )

    # Height tracking — meaningful once knees are unlocked; harmless before
    track_base_height_exp = RewardTermCfg(
        func=legged_mdp.rewards.track_base_height_exp,
        weight=1.0,
        params={"command_name": "height_command", "std": 0.05},
    )

    # Per-joint penalties — frozen joints stay at init_pos → deviation ≈ 0 → penalty ≈ 0
    joint_deviation_knee = RewardTermCfg(
        func=rewards.joint_deviation_l1,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )

    joint_deviation_thigh = RewardTermCfg(
        func=rewards.joint_deviation_l1,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint"])},
    )

    joint_deviation_hip = RewardTermCfg(
        func=rewards.joint_deviation_l1,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    dof_pos_limits = RewardTermCfg(
        func=rewards.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_hip_joint", ".*_thigh_joint", ".*_knee_joint",
        ])},
    )

    joint_acc = RewardTermCfg(
        func=rewards.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_hip_joint", ".*_thigh_joint", ".*_knee_joint",
        ])},
    )

    stand_still = RewardTermCfg(
        func=mdp_vel.stand_still_joint_deviation_l1,
        weight=-2.0,
        params={
            "command_name": "velocity_command",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_hip_joint", ".*_thigh_joint", ".*_knee_joint",
            ]),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Terminations
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumTerminationsCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

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

    illegal_contact_base = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,
            "sensor_cfg": SceneEntityCfg(name="contact_forces_base", body_names=["base"]),
        },
    )

    illegal_contact_right_leg = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_right_leg",
                body_names=["thigh", "right_hip", "right_calf_motor"],
            ),
        },
    )

    illegal_contact_left_leg = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_left_leg",
                body_names=["thigh", "left_hip", "left_calf_motor"],
            ),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum — auto joint unlock
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CurriculumCfg:
    """Curriculum: unlock joints progressively during one training run."""

    joint_unlock = CurriculumTermCfg(
        func=unlock_joint_phases,
        params={},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Env Config
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class LeggedV3CurriculumEnvCfg(ManagerBasedRLEnvCfg):
    """Single-run auto-curriculum for legged_v3 (wheel → knee → thigh → hip)."""

    scene:        CurriculumSceneCfg       = CurriculumSceneCfg(num_envs=1, env_spacing=2.0)
    observations: CurriculumObsCfg         = CurriculumObsCfg()
    actions:      CurriculumActionCfg      = CurriculumActionCfg()
    commands:     CurriculumCommandsCfg    = CurriculumCommandsCfg()
    events:       CurriculumEventCfg       = CurriculumEventCfg()
    rewards:      CurriculumRewardCfg      = CurriculumRewardCfg()
    terminations: CurriculumTerminationsCfg = CurriculumTerminationsCfg()
    curriculum:   CurriculumCfg            = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2           # control @ 30 Hz (sim 60 Hz / 2)
        self.episode_length_s = 60.0

        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
