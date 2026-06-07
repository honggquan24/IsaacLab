"""Biped-Outer-Angle — Train vòng NGOÀI của cascade (bước 2/2).

Yêu cầu: đã train Biped-Inner-Vel và có file model_*.pt.

RL → [kp_θ, ki_θ, kd_θ]  →  outer PID  →  ω_des
                           →  inner policy (frozen)  →  torque

Nhiệm vụ: cân bằng robot thẳng đứng (con lắc ngược trên bánh xe).
Vòng ngoài tính ω_des từ góc nghiêng; vòng trong (pretrained) điều khiển
tốc độ bánh xe để đạt ω_des.

Obs (11-dim): projected_gravity(3) + ang_vel_b(3) + wheel_vel(2) + last_action(3)
Action (3-dim): [kp_θ_raw, ki_θ_raw, kd_θ_raw] ∈ [-1, 1]

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Biped-Outer-Angle \\
        --num_envs 256 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Biped-Outer-Angle --num_envs 4
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
class OuterAngleSceneCfg(InteractiveSceneCfg):

    num_envs: int          = 256
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
            static_friction=0.85,
            dynamic_friction=0.65,
        ),
        debug_vis=False,
    )

    robot: Articulation = BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """RL tunes 3 outer PID gains [kp_θ, ki_θ, kd_θ].

    Gán pretrained_inner_policy_path trước khi train:
        cfg.actions.balance_pid.pretrained_inner_policy_path = "path/to/model.pt"
    """

    balance_pid = mdp.BalanceCascadeActionCfg(
        asset_name="robot",
        pretrained_inner_policy_path="rl_control/model/inner_vel/model.pt",
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Obs 11-dim: projected_gravity(3) + ang_vel_b(3) + wheel_vel(2) + last_action(3)."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):

        projected_gravity = ObservationTermCfg(func=mdp.body_projected_gravity)
        ang_vel_b         = ObservationTermCfg(func=mdp.body_ang_vel_b)
        wheel_vel         = ObservationTermCfg(func=mdp.wheel_angular_velocity)
        last_action       = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────── Events ───────────────────────────────────────────

@configclass
class EventCfg:
    """Reset: tilt ngẫu nhiên nhỏ → tạo disturbance step cho vòng ngoài."""

    reset_robot = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x":     (-0.05, 0.05),
                "y":     (-0.05, 0.05),
                "z":     (0.0,   0.0),
                "roll":  (-0.15, 0.15),   # ±8.6° initial tilt → balance step response
                "pitch": (-0.15, 0.15),
                "yaw":   (-math.pi, math.pi),
            },
            "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
        },
    )

    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.02, 0.02),
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────

@configclass
class RewardCfg:
    """Chất lượng cân bằng — step response của vòng ngoài."""

    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-500.0)

    # Primary balance
    balance_exp = RewardTermCfg(
        func=mdp.rewards.balance_exp,
        weight=10.0,
        params={"std": 0.1},
    )
    balance_l2 = RewardTermCfg(
        func=mdp.rewards.balance_l2,
        weight=-2.0,
    )

    # Step response criteria
    settling_bonus = RewardTermCfg(
        func=mdp.rewards.settling_bonus,
        weight=3.0,
        params={"band_rad": 0.03},
    )
    overshoot_penalty = RewardTermCfg(
        func=mdp.rewards.overshoot_penalty,
        weight=-5.0,
    )
    oscillation_penalty = RewardTermCfg(
        func=mdp.rewards.oscillation_penalty,
        weight=-2.0,
        params={"near_upright_rad": 0.15},
    )

    # Smoothness
    wheel_vel_l2  = RewardTermCfg(func=mdp.rewards.wheel_vel_l2,   weight=-1e-3)
    torque_l2     = RewardTermCfg(func=mdp.rewards.torque_l2,      weight=-1e-5)
    action_rate   = RewardTermCfg(func=mdp.rewards.action_rate_l2, weight=-0.01)


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={"limit_angle": math.pi / 5, "asset_cfg": SceneEntityCfg("robot")},
    )

    fallen = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={"minimum_height": 0.10, "asset_cfg": SceneEntityCfg("robot")},
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class BipedOuterAngleEnvCfg(ManagerBasedRLEnvCfg):
    """Env cho Biped-Outer-Angle (bước 2 trong cascade).

    Sim: 200 Hz. Policy (outer): 50 Hz (decimation=4).
    Inner policy chạy bên trong apply_actions() ở 200 Hz. Episode: 20 s.
    """

    scene:        OuterAngleSceneCfg = OuterAngleSceneCfg(num_envs=256, env_spacing=2.0)
    observations: ObservationsCfg    = ObservationsCfg()
    actions:      ActionCfg          = ActionCfg()
    events:       EventCfg           = EventCfg()
    rewards:      RewardCfg          = RewardCfg()
    terminations: TerminationsCfg    = TerminationsCfg()

    def __post_init__(self):
        self.decimation       = 4
        self.episode_length_s = 20.0
        self.sim.dt           = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.sim.physx.enable_external_forces_every_iteration = True
