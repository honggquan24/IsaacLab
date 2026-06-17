"""Wheeled locomotion environment cho robot V5 — clone từ V0, đổi robot + joint names.

Body names trong USD (từ Stage panel):
  base                           ← thân chính
  hip, hip_01                    ← right crank A + B
  hip_02, hip_03                 ← left crank A + B
  knee, knee_01                  ← right coupler A + B
  knee_02, knee_03               ← left coupler A + B
  wheel                          ← right wheel
  wheel_01                       ← left wheel

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V5-Wheel --num_envs 4096 --headless
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
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
from isaaclab.envs.mdp import actions, events, observations, rewards, terminations, commands

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_vel

from ..legged_v5_cfg import LEGGED_V5_CFG
from .. import mdp


# ─────────────────────────── Scene ────────────────────────────────────────────

@configclass
class LeggedV5SceneCfg(InteractiveSceneCfg):

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

    robot: Articulation = LEGGED_V5_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    contact_forces_hip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/Robot/hip.*",
        history_length=3,
        track_air_time=False,
    )

    contact_forces_knee = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/Robot/knee.*",
        history_length=3,
        track_air_time=False,
    )


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:

    # Mimic: policy chỉ ra 2 lệnh (right_hip, left_hip); mimic tự = -active (gearing -1).
    # Giảm action chân 4→2 + đảm bảo ràng buộc 5-bar luôn đúng.
    leg_pos = mdp.actions.HipMimicPositionActionCfg(
        asset_name="robot",
        active_joint_names=["right_hip_joint", "left_hip_joint"],
        mimic_joint_names=["right_hip_joint_mimic", "left_hip_joint_mimic"],
        scale=1.0,   # HẠ từ 1.0 → ±0.5 rad ≈ ±28°: chặn robot gập chân sâu tới mức knee/hip quét đất
                     # (illegal_contact ~100%). Dải này vẫn dư để chỉnh cao độ 0.25–0.35 m.
    )

    wheel_vel = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["right_wheel_joint", "left_wheel_joint"],
        scale=10.0,
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:

    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.02,   
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),       # ngang — không điều khiển
            lin_vel_y=(-0.5, 0.5),      # tiến/lùi ±0.5 m/s (FIX: giá trị trước là ±5.0 — sai 10× so với comment, khiến lệnh bất khả thi → robot bỏ bám tốc, chỉ đứng cân bằng)
            ang_vel_z=(-0.3, 0.3),      # xoay ±0.5 rad/s (FIX: trước là ±5.0)
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
            pos_z=(0.30, 0.30),
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
        base_lin_vel      = ObservationTermCfg(func=observations.base_lin_vel)
        projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
        joint_pos    = ObservationTermCfg(func=observations.joint_pos)
        joint_vel    = ObservationTermCfg(func=observations.joint_vel)
        last_action  = ObservationTermCfg(func=observations.last_action)
        velocity_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
        )
        height_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "height_command"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        root_pos_w        = ObservationTermCfg(func=observations.root_pos_w)
        root_quat_w       = ObservationTermCfg(func=observations.root_quat_w)
        root_lin_vel_w    = ObservationTermCfg(func=observations.root_lin_vel_w)
        root_ang_vel_w    = ObservationTermCfg(func=observations.root_ang_vel_w)
        base_lin_vel      = ObservationTermCfg(func=observations.base_lin_vel)
        base_ang_vel      = ObservationTermCfg(func=observations.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
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
        root_rpy_deg   = ObservationTermCfg(func=mdp.observations.root_rpy_deg)

        def __post_init__(self):
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
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
            },
        },
    )

    # ── TẠM COMMENT: domain randomization + push (bật lại khi cần robust/sim-to-real) ──
    # # Ngẫu nhiên hệ số ma sát các body robot (gồm bánh) → policy không dựa vào 1 mức bám cố định.
    # physics_material = EventTermCfg(
    #     func=events.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (0.5, 1.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )
    #
    # # Thêm/bớt khối lượng thân — operation="scale" để TỈ LỆ theo khối lượng thật (V5 nhỏ).
    # add_base_mass = EventTermCfg(
    #     func=events.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )
    #
    # # Dời tâm khối thân → robot học cân bằng với CoM lệch.
    # base_com = EventTermCfg(
    #     func=events.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.01, 0.01)},
    #     },
    # )
    #
    # # Đẩy ngẫu nhiên giữa episode → ép policy chủ động dùng bánh giữ thăng bằng.
    # push_robot = EventTermCfg(
    #     func=events.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(8.0, 12.0),
    #     params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.3, 0.3)}},
    # )
    #
    # # Khi bật push/randomization, đổi reset_position.pose_range.yaw -> (-3.14, 3.14)
    # # và velocity_range -> x(-0.2,0.2) y(-0.1,0.1) yaw(-0.2,0.2) cho khớp.


# ─────────────────────────── Rewards ──────────────────────────────────────────

@configclass
class RewardCfg:

    termination_penalty = RewardTermCfg(func=rewards.is_terminated, weight=-200.0)

    # Height tracking — TƯƠNG ĐỐI so với trục bánh (terrain-independent, đúng cho robot bánh)
    track_height = RewardTermCfg(
        func=mdp.rewards.track_base_height_above_axle_exp,
        weight=3.0,
        params={
            "command_name": "height_command",
            "std": 0.05,
            "wheel_right_body": "wheel",     # body bánh V5 (KHÔNG phải wheel_link_right của V0)
            "wheel_left_body":  "wheel_01",
        },
    )

    # Tách track_lin_vel_xy → X và Y riêng để debug từng trục trên log.
    track_lin_vel_x_exp = RewardTermCfg(
        func=mdp.rewards.track_lin_vel_x_yaw_frame_exp,
        weight=0.5,   # X bị khoá (lin_vel_x=0) → chủ yếu chống trôi ngang
        params={"command_name": "velocity_command", "std": 0.35},
    )
    track_lin_vel_y_exp = RewardTermCfg(
        func=mdp.rewards.track_lin_vel_y_yaw_frame_exp,
        weight=1.5,   # Y = hướng tiến → tín hiệu bám tốc chính (std 1.0: gradient từ trạng thái đứng)
        params={"command_name": "velocity_command", "std": 0.35},
    )

    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp_vel.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "velocity_command", "std": 0.35},
    )

    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.07)

    # Khử rung khớp: phạt gia tốc khớp CHÂN (không đụng bánh — bánh quay nhanh là bình thường)
    leg_joint_acc = RewardTermCfg(
        func=rewards.joint_acc_l2,
        weight=-8.0e-7,   # bù cho việc giữ hip damping thấp (1.0) — khử rung qua reward, không qua mechanical damping
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "right_hip_joint", "right_hip_joint_mimic",
            "left_hip_joint",  "left_hip_joint_mimic",
        ])},
    )

    # Phạt nghiêng NGANG (lateral) — KHÔNG phạt lean dọc fore-aft (cần để chạy theo Y).
    # axis=0 (body-X) là trục nghiêng ngang, đã verify bằng probe_gravity_v5.py.
    lateral_tilt = RewardTermCfg(
        func=mdp.rewards.lateral_tilt_penalty,
        weight=-50.0,
        params={"axis": 0},
    )

    # Ghì pitch (fore-aft, axis=2) NHẸ — tín hiệu chống lật dọc còn thiếu (thủ phạm bad_orientation ~49%).
    # weight nhỏ (-5, KHÔNG -50 như lateral) để robot VẪN lean được khi tăng/giảm tốc, chỉ phạt khi ngả quá.
    fore_aft_tilt = RewardTermCfg(
        func=mdp.rewards.lateral_tilt_penalty,
        weight=-12.0,
        params={"axis": 2},
    )

    com_wheel_plane = RewardTermCfg(
        func=mdp.rewards.com_wheel_plane_alignment,
        weight=-25.0,
        params={
            "wheel_right_body": "wheel",
            "wheel_left_body":  "wheel_01",
        },
    )

    # (BỎ velocity_settling) — bonus "đứng vững" bị robot farm: env đứng yên/lệnh≈0 vẫn
    # ăn điểm nên không bị ép bám lệnh tốc độ. Đã gỡ để chống reward farming (bad_orientation ~56%).

    # Phạt khi vọt quá setpoint vận tốc (dao động)
    velocity_overshoot = RewardTermCfg(
        func=mdp.rewards.velocity_overshoot_penalty,
        weight=-5.0,
        params={"command_name": "velocity_command"},
    )

    # Phạt 2 chân lệch nhau (đối xứng trái-phải) — trực tiếp cho vụ "2 chân không đều"
    hip_symmetry = RewardTermCfg(
        func=mdp.rewards.hip_symmetry_l2,
        weight=-10.0,
    )

    # Khi lệnh vận tốc ≈ 0 → phạt hip lệch khỏi tư thế mặc định (giữ 2 chân đều, đứng yên)
    stand_still = RewardTermCfg(
        func=mdp_vel.stand_still_joint_deviation_l1,
        weight=-2.0,
        params={
            "command_name": "velocity_command",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_hip_joint", "right_hip_joint_mimic",
                "left_hip_joint",  "left_hip_joint_mimic",
            ]),
        },
    )


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=mdp.terminations.bad_orientation_from_default,
        params={
            "limit_angle": 1.1,  # ~63° — đủ sớm hơn π/2 nhưng CHỪA biên lean fore-aft khi tăng tốc (0.8 quá chặt, cắt luôn maneuver hợp lệ → bỏ bám tốc)
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 150.0,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.25,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


    illegal_contact_hip = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 200.0,
            "sensor_cfg": SceneEntityCfg(name="contact_forces_hip"),
        },
    )

    illegal_contact_knee = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={
            "threshold": 200.0,
            "sensor_cfg": SceneEntityCfg(name="contact_forces_knee"),
        },
    )




# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV5WheelEnvCfg(ManagerBasedRLEnvCfg):

    scene:        LeggedV5SceneCfg = LeggedV5SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionCfg        = ActionCfg()
    commands:     CommandsCfg      = CommandsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardCfg        = RewardCfg()
    terminations: TerminationsCfg  = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 60.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
