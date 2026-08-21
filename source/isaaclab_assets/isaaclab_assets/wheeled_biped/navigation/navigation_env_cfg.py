# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical navigation environment cho robot bipedal wheel (V5).

High-level policy: xuất velocity command (vx, vy, omega) → đẩy vào policy
locomotion V5 đã train (chạy bên trong PreTrainedPolicyAction).
Low-level policy: bám velocity command (train bằng Isaac-Wheeled-Biped-Wheel).

⚠️ TRƯỚC KHI TRAIN: phải set `policy_path` tới checkpoint locomotion đã export
(logs/rsl_rl/legged_v5_wheel_mimic/<run>/exported/policy.pt).

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Wheeled-Biped-Navigation --num_envs 1024 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Wheeled-Biped-Navigation --num_envs 4
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import commands, events, observations, rewards, terminations
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..wheeled_biped_cfg import WHEELED_BIPED_CFG

# Marker đích = CHẤM ĐỎ (thay mũi tên mặc định của UniformPose2dCommand).
GOAL_DOT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_position",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)
from .. import mdp
from ..locomotion.wheel_env_cfg import ActionCfg as LowLevelActionCfg
from . import rewards as nav_rewards
from .commands import HideOnReachPose2dCommandCfg
from .pre_trained_policy_action import PreTrainedPolicyActionCfg

# ───────────────────── Low-level observation (PHẢI mirror PolicyCfg locomotion V5) ──
# Thứ tự + nội dung phải khớp đúng obs lúc train Isaac-Wheeled-Biped-Wheel.
# PreTrainedPolicyAction remap `last_action` và `velocity_cmd` lúc runtime.


@configclass
class LowLevelObsCfg(ObservationGroupCfg):
    base_lin_vel = ObservationTermCfg(func=observations.base_lin_vel)
    projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
    joint_pos = ObservationTermCfg(func=observations.joint_pos)
    joint_vel = ObservationTermCfg(func=observations.joint_vel)
    last_action = ObservationTermCfg(func=observations.last_action)  # remap runtime
    velocity_cmd = ObservationTermCfg(  # remap runtime
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


# ─────────────────────────── Scene ────────────────────────────────────────────


@configclass
class NavSceneCfg(InteractiveSceneCfg):
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

    robot: Articulation = WHEELED_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ─────────────────────────── Actions ──────────────────────────────────────────


@configclass
class NavActionCfg:
    """High-level xuất velocity command → đẩy vào policy locomotion đã train."""

    locomotion = PreTrainedPolicyActionCfg(
        asset_name="robot",
        # locomotion 2000 iter đã export
        policy_path="logs/rsl_rl/legged_v5_wheel_mimic/2026-06-17_10-06-40/exported/policy.pt",
        low_level_decimation=2,  # khớp decimation=2 lúc train locomotion (100 Hz)
        low_level_actions=LowLevelActionCfg(),  # leg_pos (mimic) + wheel_vel của V5
        low_level_observations=LowLevelObsCfg(),
        # KHỚP ĐÚNG dải train low-level: vy ±0.5, wz ±0.3 (low-level CHỈ train tới wz 0.3).
        # Trước để (0,1,1) → lệnh tới 1.0 (2× dải) → ngã 74%. Rồi (0,0.5,0.5) → wz vẫn vượt
        # (0.5 > 0.3 train, 1.7×) → low-level OOD trục xoay = trục gây lật ngang → ngã 60%.
        command_scale=(0.0, 0.5, 0.3),
        debug_vis=False,  # tắt mũi tên vận tốc (bị xoay 90°X chĩa lên trời, gây hiểu nhầm) — chỉ giữ chấm đỏ đích
    )


# ─────────────────────────── Commands ─────────────────────────────────────────


@configclass
class NavCommandsCfg:
    """Goal position (x, y world + heading) + height cố định cho low-level."""

    goal = HideOnReachPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 40.0),
        simple_heading=True,  # heading mục tiêu = hướng tới đích (bắt buộc set)
        debug_vis=True,
        marker_height=0.35,  # chấm tròn ngang tầm thân robot
        resample_on_reach=True,  # tới đích (0.3m) → đổi đích mới ngay: chấm cũ biến mất, chấm mới hiện chỗ khác
        # CURRICULUM STAGE-1 (test chẩn đoán): đích GẦN ~1m. error_pos_2d dính ~2.0 suốt
        # 140 iter ⇒ ở 2m gradient thưởng ≈0 (progress dao động quanh 0, goal_reached≈0).
        # Thu nhỏ để robot THƯỜNG chạm đích (ăn +25) và progress có hướng rõ. Nếu học được
        # ⇒ logic thưởng ĐÚNG, nới dần (-2→-3→-5) thành curriculum. Nếu vẫn không chạm 1m
        # ⇒ vấn đề sâu hơn (low-level bám lệnh / obs đích). Khôi phục: (-5,5).
        ranges=commands.UniformPose2dCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
            heading=(-3.14159, 3.14159),
        ),
        goal_pose_visualizer_cfg=GOAL_DOT_MARKER_CFG,  # đích = chấm đỏ, KHÔNG mũi tên
    )

    # velocity_command bị remap hoàn toàn (PreTrainedPolicyAction), không resample.
    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

    # Height giữ cố định = chiều cao đứng (khớp lúc train low-level).
    height_command = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="base",
        resampling_time_range=(1e9, 1e9),
        make_quat_unique=False,
        debug_vis=False,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.30, 0.30),  # = 0.30 khớp ĐÚNG lúc train low-level
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ─────────────────────────── Observations (high-level) ────────────────────────


@configclass
class NavObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        goal_pos_b = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "goal"},
        )
        root_lin_vel_b = ObservationTermCfg(func=observations.base_lin_vel)
        root_ang_vel_w = ObservationTermCfg(func=observations.root_ang_vel_w)
        projected_gravity = ObservationTermCfg(func=observations.projected_gravity)
        last_action = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────── Events ───────────────────────────────────────────


@configclass
class NavEventCfg:
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
                # CURRICULUM STAGE-1: spawn gần gốc để robot→đích ≈1m (khớp dải đích ±1). BỎ yaw: base xoay 90°X nên yaw
                # làm robot LẬT. Khôi phục: (-3,3)
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────


@configclass
class NavRewardCfg:
    termination_penalty = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-10.0,
        # hạ từ -200: -200 tạo spike phương sai khổng lồ → value loss nổ, noise_std phình. Tầng cao không cần phạt nặng
        # vụ ngã (việc của low-level).
    )

    # weight 5->10: progress là tín hiệu DÀY DUY NHẤT thưởng việc lái tới đích (không farm được).
    # Tăng để khi robot tình cờ đi đúng (nhờ entropy cao), gradient đủ mạnh giữ hành vi đó.
    progress = RewardTermCfg(
        func=nav_rewards.progress_toward_goal,
        weight=10.0,
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
    )
    # CHỐNG FARM ĐỨNG YÊN: std 2.0→0.5 (thu hẹp vùng thưởng) + weight 2.0→1.0.
    # Cũ: ở cách đích 2 m vẫn được exp(-2/2)*2 ≈ 0.74/bước "miễn phí" → robot đứng giữa ăn điểm.
    # Mới: ở 2 m chỉ còn exp(-2/0.5)*1 ≈ 0.018 ≈ 0 → goal_proximity thành BONUS TIẾP CẬN sát đích
    # (<~0.5 m), không còn farm được khi đứng xa. Đường đi do `progress` (w=5) dẫn dắt.
    goal_proximity = RewardTermCfg(
        func=nav_rewards.goal_distance_exp,
        weight=1.0,
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot"), "std": 0.5},
    )
    goal_reached = RewardTermCfg(
        func=nav_rewards.goal_reached,
        weight=25.0,  # hạ từ 50: giảm spike thưa cho critic dễ fit
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot"), "threshold": 0.3},
    )
    heading = RewardTermCfg(
        func=nav_rewards.heading_toward_goal,
        weight=1.0,
        params={"command_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.05)
    lin_vel_z_l2 = RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-0.5)


# ─────────────────────────── Terminations ─────────────────────────────────────


@configclass
class NavTermCfg:
    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=mdp.terminations.bad_orientation_from_default,
        params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg(name="robot")},
    )


# ─────────────────────────── Env ──────────────────────────────────────────────


@configclass
class WheeledBipedNavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene: NavSceneCfg = NavSceneCfg(num_envs=1, env_spacing=8.0)
    observations: NavObsCfg = NavObsCfg()
    actions: NavActionCfg = NavActionCfg()
    commands: NavCommandsCfg = NavCommandsCfg()
    events: NavEventCfg = NavEventCfg()
    rewards: NavRewardCfg = NavRewardCfg()
    terminations: NavTermCfg = NavTermCfg()

    def __post_init__(self):
        self.decimation = 20  # high-level @ 10 Hz (sim 200 / 20)
        self.episode_length_s = 60.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
