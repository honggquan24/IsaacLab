"""Velocity tracking environment for Legged Robot V3.

Robot: 2-legged wheeled robot (robot_legged_v2 from Onshape)
Task: Track linear and angular velocity commands while maintaining balance.

Reward design inspired by H1RoughEnvCfg:
- Exponential velocity tracking rewards (better shaped than L2 penalties)
- Joint deviation penalty to keep legs in stable default stance
- Joint position limits penalty
- Vertical velocity penalty (no bouncing)
- Stand-still leg stabilization when command ~ 0
- IMU-based upright/balance reward

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Velocity \\
        --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V3-Velocity \\
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

# Velocity mdp: track_lin_vel_xy_yaw_frame_exp, track_ang_vel_z_world_exp,
#               stand_still_joint_deviation_l1, joint_deviation_l1, joint_pos_limits
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_vel

from .legged_v3_cfg import LEGGED_ROBOT_V3_CFG
from . import mdp


# ─────────────────────────── Scene ────────────────────────────────────────────

@configclass
class LeggedV3SceneCfg(InteractiveSceneCfg):
    """Scene for legged_v3 velocity tracking."""

    num_envs: int = 1

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    cfg_ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: Articulation = LEGGED_ROBOT_V3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # IMU on base link.
    # NOTE: Adjust prim_path after exporting USD from Onshape.
    # Check USD structure with: print(env.scene["robot"].data.body_names)
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/base",
        update_period=0.02,  # 50 Hz
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )

    # USD hierarchy:
    #   robot_legged_v3/robot_legged_v3/
    #     ├── base
    #     ├── right_leg/ → thigh, right_hip, right_calf_motor, right_wheel
    #     └── left_leg/  → thigh, left_hip,  left_calf_motor,  left_wheel
    #
    # NOTE: KHÔNG dùng ".*_leg/.*" cho cả 2 leg trong 1 sensor.
    #   IsaacLab build regex body_names từ leg đầu tiên khớp (right_leg),
    #   nhưng left_leg có tên khác (left_hip ≠ right_hip) → count mismatch → RuntimeError.
    #   Giải pháp: 3 sensor riêng biệt.

    # Sensor 1: base
    contact_forces_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/base",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # Sensor 2: right_leg (thigh, right_hip, right_calf_motor, right_wheel)
    contact_forces_right_leg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/right_leg/.*",
        update_period=0.0,
        # history_length=3,
        debug_vis=False,
    )

    # Sensor 3: left_leg (thigh, left_hip, left_calf_motor, left_wheel)
    contact_forces_left_leg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v3/robot_legged_v3/left_leg/.*",
        update_period=0.0,
        # history_length=3,
        debug_vis=False,
    )


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """Joint effort control for all 8 DOF."""

    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_joint",    # hip abduction L
            "left_thigh_joint",  # thigh pitch L
            "left_knee_joint",   # knee L
            "left_wheel_joint",  # wheel L (differential drive)
            "right_hip_joint",   # hip abduction R
            "right_thigh_joint", # thigh pitch R
            "right_knee_joint",  # knee R
            "right_wheel_joint", # wheel R (differential drive)
        ],
        scale={
            "left_hip_joint":    10.0,
            "left_thigh_joint":  10.0,
            "left_knee_joint":   10.0,
            "left_wheel_joint":  10.0,
            "right_hip_joint":   10.0,
            "right_thigh_joint": 10.0,
            "right_knee_joint":  10.0,
            "right_wheel_joint": 10.0,
        },
        debug_vis=False,
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:
    """Velocity commands for differential drive (x forward, z turn)."""

    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.5,    # 20% of envs stand still → balance training
        heading_command=False,    # angular velocity mode (not heading angle)
        debug_vis=True,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # forward / backward  [m/s]
            lin_vel_y=(0.0, 0.0),   # no lateral (differential drive)
            ang_vel_z=(-0.5, 0.5),  # turn rate  [rad/s]
            heading=(0.0, 0.0),     # unused
        ),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Observations for policy and critic."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observations (onboard sensors only — deployable on real robot)."""

        # IMU
        imu_lin_acc          = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel          = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation      = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)

        # Joint states
        joint_pos  = ObservationTermCfg(func=observations.joint_pos)
        joint_vel  = ObservationTermCfg(func=observations.joint_vel)

        # Last action (for temporal smoothness)
        last_action = ObservationTermCfg(func=observations.last_action)

        # Velocity command
        velocity_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
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

        imu_lin_acc          = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel          = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation      = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)

        joint_pos   = ObservationTermCfg(func=observations.joint_pos)
        joint_vel   = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)

        last_action = ObservationTermCfg(func=observations.last_action)

        velocity_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
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
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x":     (-0.5, 0.5),
                "y":     (-0.5, 0.5),
                "z":     (0.28, 0.32),
                "roll":  (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw":   (-math.pi, math.pi),
            },
            "velocity_range": {
                "linear":  (-0.05, 0.05),
                "angular": (-0.05, 0.05),
            },
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────

# Leg joint names (used in multiple reward terms)
_LEG_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_knee_joint"]

@configclass
class RewardCfg:
    """Reward terms — inspired by H1RoughEnvCfg.

    Positive rewards (range 0–1, exp-kernel):
        track_lin_vel_xy_exp  — forward/lateral velocity tracking
        track_ang_vel_z_exp   — yaw rate tracking
        upright               — roll/pitch alignment (balance)

    Penalties (negative):
        termination_penalty   — heavy penalty on episode failure
        lin_vel_z_l2          — suppress vertical bouncing
        joint_deviation_leg   — keep leg joints near default stance
        dof_pos_limits        — penalize joint limit violations
        stand_still           — stabilize legs when velocity command ≈ 0
        action_rate           — penalize rapid action changes (smoothness)
    """

    # ── Termination ──────────────────────────────────────────────────────────
    termination_penalty = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-200.0,
    )

    # ── Velocity tracking (exp-kernel, à la H1) ───────────────────────────────
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp_vel.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp_vel.track_ang_vel_z_world_exp,
        weight=1,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    # ── Balance: keep robot upright (exp-kernel, range [0,1]) ─────────────────
    # upright = RewardTermCfg(
    #     func=mdp.rewards.rpy_alignment_imu,
    #     weight=1.0,
    #     params={
    #         "target_rpy": (0.0, 0.0, 0.0),
    #         "imu_cfg": SceneEntityCfg(name="imu"),
    #     },
    # )

    # ── Suppress vertical velocity (no bouncing) ──────────────────────────────
    lin_vel_z_l2 = RewardTermCfg(
        func=rewards.lin_vel_z_l2,
        weight=-0.5,
    )

    # ── Keep leg joints near default stance (à la H1 joint_deviation_hip) ─────
    # joint_deviation_leg = RewardTermCfg(
    #     func=rewards.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS),
    #     },
    # )

    # ── Penalize joint limit violations (requires limits in USD) ──────────────
    dof_pos_limits = RewardTermCfg(
        func=rewards.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS),
        },
    )

    # ── When command ≈ 0, penalize deviation of leg joints from default ────────
    stand_still = RewardTermCfg(
        func=mdp_vel.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "velocity_command",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS),
        },
    )

    # ── Action smoothness ─────────────────────────────────────────────────────
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.001,
    )


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    # Fallen over (> 90 deg tilt)
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 2,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # Robot collapsed (wheels dragging on ground)
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

    # Terminate khi base chạm mặt đất → robot ngã hẳn.

    illegal_contact_base = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,   # [N]
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_base",
                body_names=["base"],
            ),
        },
    )

    # Terminate khi right_leg (trừ bánh xe) chạm đất.
    # body_names: thigh, right_hip, right_calf_motor — loại trừ right_wheel.
    illegal_contact_right_leg = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,   # [N]
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_right_leg",
                body_names=["thigh", "right_hip", "right_calf_motor"],
            ),
        },
    )

    # Terminate khi left_leg (trừ bánh xe) chạm đất.
    # body_names: thigh, left_hip, left_calf_motor — loại trừ left_wheel.
    illegal_contact_left_leg = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 50.0,   # [N]
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces_left_leg",
                body_names=["thigh", "left_hip", "left_calf_motor"],
            ),
        },
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV3VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Environment config for legged_v3 velocity tracking."""

    scene:        LeggedV3SceneCfg  = LeggedV3SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionCfg        = ActionCfg()
    commands:     CommandsCfg      = CommandsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardCfg        = RewardCfg()
    terminations: TerminationsCfg  = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2           # control @ 30 Hz  (sim 60 Hz / 2)
        self.episode_length_s = 10.0

        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
