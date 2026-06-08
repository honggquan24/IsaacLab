"""Biped-Outer-Vel-PID — Outer loop: velocity command → velocity PID → tilt setpoint → inner.

Kiến trúc:
    vel_cmd (vx, vy, yaw_rate)
        └─ Outer RL (9 gains) ──► Velocity PID ──► (roll_des, pitch_des, yaw_des)
                                                         └─ Inner TiltPIDAction (fixed gains)
                                                                └─ torque

RL outputs 9 gains: [[kp,ki,kd], [kp,ki,kd], [kp,ki,kd]]
  PID 0: vy_err  → roll_des   (lean forward/back để đạt tốc độ)
  PID 1: vx_err  → pitch_des  (lateral)
  PID 2: yaw_rate_err → yaw_rate_cmd → integrate → yaw_des

Inner TiltPIDAction chạy với reference gains (không RL-tune).
Outer ghi đè target_tilt command mỗi bước.

Obs (23-dim):
    velocity_command(3) + velocity_error(3) + base_lin_vel_b(3)
    + ang_vel_b(3) + projected_gravity(3) + wheel_vel(2) + last_action(9) - 1 (pad)
    = thực tế 26-dim (xem ObservationsCfg)

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Outer-Vel-PID --num_envs 1024 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Biped-Outer-Vel-PID --num_envs 4
"""
from __future__ import annotations
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import observations, events, rewards, terminations

from isaaclab_assets.biped.biped_cfg import BIPED_CFG
from isaaclab_assets.biped import mdp


# ─────────────────────────── Scene ────────────────────────────────────────────

@configclass
class OuterVelPIDSceneCfg(InteractiveSceneCfg):

    num_envs: int           = 512
    replicate_physics: bool = True

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1200.0),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
        ),
        debug_vis=False,
    )

    robot: Articulation = BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """9 outer gains + embedded inner TiltPIDAction (fixed reference gains)."""

    outer_pid = mdp.OuterVelPIDActionCfg(
        vel_command_name="velocity_cmd",
        tilt_command_name="target_tilt",
        action_scale=2.0,
        inner_cfg=mdp.TiltPIDActionCfg(
            asset_name="robot",
            action_scale=0.0,           # inner gains frozen at bias
            command_name="target_tilt",
        ),
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:

    # Velocity setpoint — outer RL's tracking objective
    velocity_cmd = mdp.VelocityCommandCfg(
        resampling_time_range=(4.0, 8.0),
        vx_range=(-0.3, 0.3),
        vy_range=(-0.5, 0.5),
        yaw_rate_range=(-1.0, 1.0),
    )

    # Tilt setpoint — written by outer action each step, read by embedded inner
    target_tilt = mdp.TargetTiltCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),   # never auto-resample; outer owns it
        roll_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        yaw_delta_range=(0.0, 0.0),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Obs 26-dim:
    velocity_command(3) + velocity_error(3) + base_lin_vel_b(3)
    + ang_vel_b(3) + projected_gravity(3) + wheel_vel(2) + last_action(9)
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):

        velocity_command  = ObservationTermCfg(
            func=mdp.velocity_command,
            params={"command_name": "velocity_cmd"},
        )
        velocity_error    = ObservationTermCfg(
            func=mdp.velocity_error,
            params={"command_name": "velocity_cmd"},
        )
        base_lin_vel_b    = ObservationTermCfg(func=mdp.base_lin_vel_b)
        ang_vel_b         = ObservationTermCfg(func=observations.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
        wheel_vel         = ObservationTermCfg(func=mdp.wheel_angular_velocity)
        last_action       = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────── Events ───────────────────────────────────────────

@configclass
class EventCfg:

    reset_robot = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x":     (-0.1, 0.1),
                "y":     (-0.1, 0.1),
                "z":     (0.0,  0.0),
                "roll":  (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw":   (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.05, 0.05),
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────

@configclass
class RewardCfg:

    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-300.0)

    # Primary: bám tốc độ
    velocity_tracking = RewardTermCfg(
        func=mdp.rewards.velocity_tracking_exp,
        weight=8.0,
        params={"command_name": "velocity_cmd", "std": 0.3},
    )
    yaw_rate_tracking = RewardTermCfg(
        func=mdp.rewards.yaw_rate_tracking_exp,
        weight=3.0,
        params={"command_name": "velocity_cmd", "std": 0.3},
    )
    lin_vel_l2 = RewardTermCfg(
        func=mdp.rewards.lin_vel_l2,
        weight=-1.0,
        params={"command_name": "velocity_cmd"},
    )

    # Secondary: giữ thẳng
    upright = RewardTermCfg(
        func=mdp.rewards.upright_exp,
        weight=2.0,
        params={"std": 0.2},
    )

    # Smoothness
    wheel_torque = RewardTermCfg(func=mdp.rewards.wheel_torque_l2, weight=-1e-4)
    action_rate  = RewardTermCfg(func=mdp.rewards.action_rate_l2,  weight=-0.01)


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={"limit_angle": math.pi / 4.5, "asset_cfg": SceneEntityCfg("robot")},
    )

    fallen = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={"minimum_height": 0.10, "asset_cfg": SceneEntityCfg("robot")},
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class BipedOuterVelPIDEnvCfg(ManagerBasedRLEnvCfg):
    """Outer loop: velocity command → velocity PID gains → tilt_cmd → inner TiltPID.

    Sim 200 Hz. Policy (outer) 50 Hz (decimation=4). Episode 20 s.
    """

    scene:        OuterVelPIDSceneCfg = OuterVelPIDSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg     = ObservationsCfg()
    actions:      ActionCfg           = ActionCfg()
    commands:     CommandsCfg         = CommandsCfg()
    events:       EventCfg            = EventCfg()
    rewards:      RewardCfg           = RewardCfg()
    terminations: TerminationsCfg     = TerminationsCfg()

    def __post_init__(self):
        self.decimation          = 4
        self.episode_length_s    = 20.0
        self.sim.dt              = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.sim.physx.enable_external_forces_every_iteration = True
