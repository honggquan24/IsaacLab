"""Biped-Inner-Tilt — Train vòng TRONG của cascade (bước 1).

RL → 7 × [kp, ki, kd] = 21 outputs:
    4 hip  position PIDs → tau_hip
    2 wheel velocity PIDs → tau_balance
    1 yaw  PID            → tau_yaw
    tau_left = tau_balance_L + tau_yaw
    tau_right= tau_balance_R - tau_yaw

Nhiệm vụ: bám cmd_tilt (roll_des, pitch_des, yaw_des) — step input ngẫu nhiên.

Obs (40-dim):
    tilt_error(3) + ang_vel_b(3) + projected_gravity(3)
    + hip_pos_error(4) + hip_vel(4) + wheel_vel(2) + last_action(21)

Action (21-dim): 7 × [kp_raw, ki_raw, kd_raw] ∈ [-1,1]

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Inner-Tilt --num_envs 512 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Biped-Inner-Tilt --num_envs 4
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
class InnerTiltSceneCfg(InteractiveSceneCfg):

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
    """21 outputs = 7 × [kp, ki, kd]."""

    tilt_pid = mdp.TiltPIDActionCfg(
        asset_name="robot",
        action_scale=5.0,
        command_name="target_tilt",
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:
    """Step input: random tilt setpoint."""

    target_tilt = mdp.TargetTiltCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        roll_range=(-0.05,  0.05),
        pitch_range=(-0.05, 0.05),
        yaw_delta_range=(-0.3, 0.3),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Obs 74-dim:
    tilt_error(3) + imu_quat(4) + imu_lin_acc_b(3) + ang_vel_b(3)
    + projected_gravity(3) + hip_pos_error(4) + hip_vel(4) + wheel_vel(2)
    + all_joint_pos(10) + all_joint_vel(10) + all_joint_acc(10) + last_action(21)
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):

        # Tilt setpoint error
        tilt_error = ObservationTermCfg(
            func=mdp.tilt_error,
            params={"command_name": "target_tilt"},
        )
        # IMU
        imu_quat      = ObservationTermCfg(func=mdp.imu_quat)
        imu_lin_acc_b = ObservationTermCfg(func=mdp.imu_lin_acc_b)
        ang_vel_b     = ObservationTermCfg(func=observations.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
        # Hip tracking error
        hip_pos_error = ObservationTermCfg(
            func=mdp.hip_pos_error,
            params={"command_name": "target_tilt"},
        )
        hip_vel   = ObservationTermCfg(func=mdp.hip_velocity)
        wheel_vel = ObservationTermCfg(func=mdp.wheel_angular_velocity)
        # Tất cả joints (10 joints)
        all_joint_pos = ObservationTermCfg(func=mdp.all_joint_pos_rel)
        all_joint_vel = ObservationTermCfg(func=mdp.all_joint_vel)
        all_joint_acc = ObservationTermCfg(func=mdp.all_joint_acc)
        # Last action
        last_action = ObservationTermCfg(func=observations.last_action)

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
                "x":     (-0.05, 0.05),
                "y":     (-0.05, 0.05),
                "z":     (0.0,   0.0),
                "roll":  (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw":   (-math.pi, math.pi),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.1,  0.1),
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────

@configclass
class RewardCfg:
    """Chất lượng bám tilt — dựa trên đặc tính biểu đồ đáp ứng."""

    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-200.0)

    # Primary tracking
    tilt_tracking_exp = RewardTermCfg(
        func=mdp.rewards.tilt_tracking_exp,
        weight=10.0,
        params={"command_name": "target_tilt", "std": 0.05},
    )
    yaw_tracking_exp = RewardTermCfg(
        func=mdp.rewards.yaw_tracking_exp,
        weight=3.0,
        params={"command_name": "target_tilt", "std": 0.1},
    )
    tilt_tracking_l2 = RewardTermCfg(
        func=mdp.rewards.tilt_tracking_l2,
        weight=-2.0,
        params={"command_name": "target_tilt"},
    )

    # Step response
    settling_bonus = RewardTermCfg(
        func=mdp.rewards.settling_bonus,
        weight=5.0,
        params={"command_name": "target_tilt", "band_roll_pitch": 0.03, "band_yaw": 0.05},
    )
    overshoot_penalty = RewardTermCfg(
        func=mdp.rewards.overshoot_penalty,
        weight=-3.0,
        params={"command_name": "target_tilt"},
    )
    oscillation_penalty = RewardTermCfg(
        func=mdp.rewards.oscillation_penalty,
        weight=-2.0,
        params={"command_name": "target_tilt", "near_band": 0.05},
    )

    # Smoothness
    hip_torque   = RewardTermCfg(func=mdp.rewards.hip_torque_l2,   weight=-1e-4)
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
class BipedInnerTiltEnvCfg(ManagerBasedRLEnvCfg):
    """Env cho Biped-Inner-Tilt.

    Sim: 200 Hz. Policy: 50 Hz (decimation=4). Episode: 15 s.
    """

    scene:        InnerTiltSceneCfg = InnerTiltSceneCfg(num_envs=512, env_spacing=2.0)
    observations: ObservationsCfg   = ObservationsCfg()
    actions:      ActionCfg         = ActionCfg()
    commands:     CommandsCfg       = CommandsCfg()
    events:       EventCfg          = EventCfg()
    rewards:      RewardCfg         = RewardCfg()
    terminations: TerminationsCfg   = TerminationsCfg()

    def __post_init__(self):
        self.decimation          = 1      # policy 200 Hz = sim rate (inner loop)
        self.episode_length_s    = 15.0
        self.sim.dt              = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.sim.physx.enable_external_forces_every_iteration = True
