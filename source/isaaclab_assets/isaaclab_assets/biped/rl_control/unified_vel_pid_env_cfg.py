"""Biped-Unified-Vel — Single-stage: vel_cmd → 7 PIDs → torque trực tiếp.

Kiến trúc:
    vel_cmd (vy, vx, yaw_rate)
        └─ RL (21 dim = 7 × [kp,ki,kd])
               └─ VelDirectPIDAction
                      ├─ Hip PID  [0-3]: vy_err × alloc → tau_hip
                      ├─ Wheel PID[4-5]: vy_err         → tau_balance
                      └─ Yaw PID  [6]  : yr_err          → tau_yaw

Không có tilt setpoint trung gian. RL học trực tiếp từ velocity error.

Obs (43-dim):
    velocity_command(3) + velocity_error(2) + base_lin_vel_b(3)
    + ang_vel_b(3) + projected_gravity(3) + wheel_vel(2)
    + hip_vel(4) + all_joint_pos(10) + last_action(21)

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Unified-Vel --num_envs 512 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Biped-Unified-Vel --num_envs 4
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
class UnifiedVelSceneCfg(InteractiveSceneCfg):

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
            static_friction=2.0,
            dynamic_friction=1.8,
        ),
        debug_vis=False,
    )

    robot: Articulation = BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """21 outputs = 7 × [kp, ki, kd], error = velocity error trực tiếp."""

    vel_pid = mdp.VelDirectPIDActionCfg(
        asset_name="robot",
        vel_command_name="velocity_cmd",
        action_scale=1.0,
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:

    velocity_cmd = mdp.VelocityCommandCfg(
        resampling_time_range=(4.0, 8.0),
        vx_range=(0.0, 0.0),
        vy_range=(0.0, 0.0),       # phase 1: balance only
        yaw_rate_range=(0.0, 0.0), # phase 1: balance only
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Obs 70-dim:
    velocity_command(3) + velocity_error(2) + base_lin_vel_b(3)
    + ang_vel_b(3) + projected_gravity(3) + wheel_vel(2)
    + hip_vel(4) + all_joint_pos(10) + all_joint_vel(10) + all_joint_acc(10) + last_action(28)
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
        hip_vel           = ObservationTermCfg(func=mdp.hip_velocity)
        all_joint_pos     = ObservationTermCfg(func=mdp.all_joint_pos_rel)
        all_joint_vel     = ObservationTermCfg(func=mdp.all_joint_vel)
        all_joint_acc     = ObservationTermCfg(func=mdp.all_joint_acc)
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
                "roll":  (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
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
    """Tiêu chí đáp ứng bước nhảy cho vận tốc (step response quality)."""

    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-300.0)

    # ── Bám setpoint ───────────────────────────────────────────────────────────
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

    # ── Đáp ứng bước nhảy ─────────────────────────────────────────────────────
    settling_bonus = RewardTermCfg(
        func=mdp.rewards.velocity_settling_bonus,
        weight=5.0,
        params={"command_name": "velocity_cmd", "band_vel": 0.05, "band_yaw": 0.1},
    )
    overshoot_penalty = RewardTermCfg(
        func=mdp.rewards.velocity_overshoot_penalty,
        weight=-3.0,
        params={"command_name": "velocity_cmd"},
    )
    # ── Đứng thẳng ────────────────────────────────────────────────────────────
    upright = RewardTermCfg(
        func=mdp.rewards.upright_exp,
        weight=10.0,
        params={"std": 0.15},
    )
    rpy_alignment = RewardTermCfg(
        func=mdp.rewards.rpy_alignment,
        weight=10.0,
        params={"std_roll": 0.08},
    )

    # ── Smoothness / Energy ────────────────────────────────────────────────────
    lin_vel_l2   = RewardTermCfg(
        func=mdp.rewards.lin_vel_l2,
        weight=-0.5,
        params={"command_name": "velocity_cmd"},
    )
    wheel_torque = RewardTermCfg(func=mdp.rewards.wheel_torque_l2, weight=-1e-4)
    hip_torque   = RewardTermCfg(func=mdp.rewards.hip_torque_l2,   weight=-1e-4)
    action_rate  = RewardTermCfg(func=mdp.rewards.action_rate_l2,  weight=-0.01)


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={"limit_angle": math.pi / 4, "asset_cfg": SceneEntityCfg("robot")},
    )

    fallen = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={"minimum_height": 0.10, "asset_cfg": SceneEntityCfg("robot")},
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class BipedUnifiedVelEnvCfg(ManagerBasedRLEnvCfg):
    """Single-stage vel→torque, policy 50 Hz, episode 20 s."""

    scene:        UnifiedVelSceneCfg = UnifiedVelSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg    = ObservationsCfg()
    actions:      ActionCfg          = ActionCfg()
    commands:     CommandsCfg        = CommandsCfg()
    events:       EventCfg           = EventCfg()
    rewards:      RewardCfg          = RewardCfg()
    terminations: TerminationsCfg    = TerminationsCfg()

    def __post_init__(self):
        self.decimation          = 2       # policy 50 Hz
        self.episode_length_s    = 20.0
        self.sim.dt              = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.sim.physx.enable_external_forces_every_iteration = True
