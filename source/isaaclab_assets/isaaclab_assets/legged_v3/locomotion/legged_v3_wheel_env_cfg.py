"""Wheeled locomotion environment for Legged Robot V3.

Robot: 2-legged wheeled robot (5-bar parallel linkage per leg)
Task: Track velocity commands using wheel-based locomotion while maintaining balance.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Legged-V3-Wheel \
        --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Legged-V3-Wheel \
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
            static_friction=2.0,
            dynamic_friction=1.8,
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

    # Contact sensor on all body links — used for illegal_contact termination.
    # Covers thigh/shin links (should never touch ground) + base_link.
    # foot_links hold the wheels and are expected to contact the ground.
    contact_forces_body = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """Position control for hip A1 joints + velocity control for wheels.

    Knee joints (B1, B2) are passive — constrained by the 5-bar loop closure.
    Hip A2 joints are mimic (follow A1 via PhysxMimicJointAPI).
    """

    # Hip A1: position control (2 DOF — left and right)
    hip_pos = actions.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_joint_A1", "right_hip_joint_A1"],
        scale=1.0,
    )

    # Wheels: velocity control (2 DOF)
    wheel_vel = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=1.0,
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:
    """Velocity + height commands."""

    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.5,
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
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
            pos_z=(0.20, 0.26),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Observation groups for policy and critic.

    Policy group uses only onboard sensors (IMU + encoders + last action + commands)
    so the trained network can be deployed on the real robot without extra infrastructure.
    Critic group adds privileged simulator state (root velocity, joint effort, episode time)
    that is available during training but not at deployment.
    """

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
        # Sai lệch vận tốc thực so với setpoint — policy biết cần tăng/giảm bao nhiêu
        # Có thể deploy được: ước lượng từ encoder bánh xe + kinematic trên robot thật
        velocity_error = ObservationTermCfg(
            func=mdp.observations.velocity_error,
            params={"command_name": "velocity_command"},
        )
        # Tốc độ góc 2 bánh xe — feedback trực tiếp từ encoder
        wheel_vel = ObservationTermCfg(func=mdp.observations.wheel_angular_velocity)

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
    """Reset events applied at episode boundaries.

    Joints are perturbed by a small offset so the policy learns to recover
    from imperfect initial poses. Root position and yaw are randomized over a
    wide area to prevent the policy from exploiting a fixed spawn location.
    """

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

_LEG_JOINTS = [".*_hip_joint_A1"]  # only active hip joints used in reward shaping

@configclass
class RewardCfg:
    """Reward shaping for wheeled balance locomotion.

    Primary task rewards (positive weights): velocity tracking (linear + angular) and
    base height tracking. Stability and smoothness penalties (negative weights) discourage
    tilting, jerky joint accelerations, high torques, and rapid action changes.
    """

    # ── Primary task ──────────────────────────────────────────────────────────
    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp_vel.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp_vel.track_ang_vel_z_world_exp,
        weight=4.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    track_base_height_l2 = RewardTermCfg(
        func=mdp.rewards.track_base_height_l2,
        weight=-20.0,
        params={"command_name": "height_command"},
    )

    # ── Stability ─────────────────────────────────────────────────────────────
    # Gaussian kernel: = 1 khi thẳng đứng, decay về 0 khi nghiêng.
    # Weight dương tạo gradient liên tục bootstrap balance từ đầu training.
    upright_exp = RewardTermCfg(
        func=mdp.rewards.upright_exp,
        weight=5.0,
        params={"std": 0.3},
    )

    # lin_vel_z_l2 = RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-1.0)

    # ── Joint / action smoothness ─────────────────────────────────────────────
    # dof_pos_limits = RewardTermCfg(
    #     func=rewards.joint_pos_limits,
    #     weight=-2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    # )

    stand_still = RewardTermCfg(
        func=mdp_vel.stand_still_joint_deviation_l1,
        weight=-1.5,
        params={
            "command_name": "velocity_command",
            "command_threshold": 0.2,
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
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )

    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.05)

    # ── Step-response quality ─────────────────────────────────────────────────
    # Bonus khi vx và yaw_rate đã ổn định trong dải sai số (settling time ngắn)
    velocity_settling = RewardTermCfg(
        func=mdp.rewards.velocity_settling_bonus,
        weight=2.0,
        params={"command_name": "velocity_command", "band_vel": 0.10, "band_yaw": 0.15},
    )

    # Penalty khi vận tốc vượt qua setpoint — giảm overshoot
    velocity_overshoot = RewardTermCfg(
        func=mdp.rewards.velocity_overshoot_penalty,
        weight=-3.0,
        params={"command_name": "velocity_command"},
    )


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:
    """Termination conditions for the wheeled locomotion task.

    Episodes end early on timeout, excessive tilt (>45°), dangerously high joint
    velocity, or the base dropping below a minimum height — a proxy for falling.
    Base contact detection is not used because the structural tilt of the chassis
    causes a corner to briefly touch the ground at q=0; bad_orientation handles
    genuine falls more robustly.
    """

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 4,
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
            "max_velocity": 60.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*_hip_joint.*", ".*_knee_joint.*"]),
        },
    )

    # Shin (knee) links must never touch ground.
    illegal_contact = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces_body",
                body_names=[".*_shin_link.*"],
            ),
        },
    )

    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.05,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV3WheelEnvCfg(ManagerBasedRLEnvCfg):
    """Full environment config for Isaac-Legged-V3-Wheel.

    Simulation runs at 200 Hz; the control policy runs at 50 Hz (decimation=4).
    Episode length is 60 s. Viewer is positioned for a side-angle overview.
    """

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
