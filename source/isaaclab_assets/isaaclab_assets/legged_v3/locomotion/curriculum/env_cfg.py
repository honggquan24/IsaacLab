"""Command-range curriculum environment for Legged Robot V3.

Robot: 2-legged wheeled robot (5-bar parallel linkage per leg)

Curriculum phases (single training run):
  Phase 0  (iter    0 –  999): Stand still + reach height.
                                Velocity commands = 0, rel_standing_envs = 1.0
  Phase 1  (iter 1000 – 2999): Slow movement ≤ 0.5 m/s.
  Phase 2  (iter 3000+):       Full speed ≤ 1.0 m/s.

All rewards, observations, and actions are identical to the base wheel env.
Only the velocity command range changes over time via CurriculumManager.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V3-Curriculum \\
        --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V3-Curriculum \\
        --num_envs 4
"""

from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import commands, rewards, actions

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_vel
from ... import mdp

from ..legged_v3_wheel_env_cfg import (
    LeggedV3SceneCfg,
    ObservationsCfg,
    EventCfg,
    TerminationsCfg,
)
from .mdp import expand_velocity_command_range

_LEG_JOINTS = ["pad_joint_.*", "thigh_joint_.*_1", "calf_joint_.*_1"]


# ─────────────────────────── Actions ─────────────────────────────────────────

@configclass
class CurriculumActionCfg:
    """Same joints as wheel env but smaller scale for Phase-0 stability.

    scale=15.0 + init_noise_std=1.0 → actions up to ±15 rad → legs snap to
    extreme angles in the first step → immediate illegal contact termination.
    scale=1.0 keeps the robot near its default pose while learning to stand.
    """

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

    wheel_vel = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_joint_right", "wheel_joint_left"],
        scale=1.0,
    )


# ─────────────────────────── Rewards (Phase-0 safe) ──────────────────────────

@configclass
class CurriculumRewardCfg:
    """Reward config for curriculum training.

    Khác wheel env:
    - stand_still bị xóa: Phase 0 cmd=0 luôn → term này active mọi lúc,
      mâu thuẫn với height tracking (robot cần chỉnh chân để đạt height).
    - joint_deviation_pad giảm -15 → -3: cho linkage tự tìm tư thế đứng.
    """

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
        weight=4.0,
        params={"command_name": "velocity_command", "std": 0.5},
    )

    # track_base_height_exp = RewardTermCfg(
    #     func=mdp.rewards.track_base_height_exp,
    #     weight=8.0,
    #     params={"command_name": "height_command", "std": 0.15},
    # )

    upright = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=-5.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
        },
    )

    joint_deviation_pad = RewardTermCfg(
        func=rewards.joint_deviation_l1,
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="pad_joint_.*")},
    )

    joint_acc = RewardTermCfg(
        func=rewards.joint_acc_l2,
        weight=-2e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )

    joint_torques = RewardTermCfg(
        func=rewards.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)},
    )

    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.5)


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CurriculumCommandsCfg:
    """Phase 0: standing only. Range expands automatically via curriculum."""

    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=1.0,          # Phase 0: all envs stand still
        heading_command=False,
        debug_vis=False,
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),       # unlocked by curriculum
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),       # unlocked by curriculum
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
            pos_z=(0.25, 0.30),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ─────────────────────────── Curriculum ───────────────────────────────────────

@configclass
class CurriculumCfg:
    """Expand velocity command range at iteration thresholds."""

    velocity_expansion = CurriculumTermCfg(
        func=expand_velocity_command_range,
        params={},
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV3CurriculumEnvCfg(ManagerBasedRLEnvCfg):
    """Curriculum env: same as wheel env but starts with standing-only commands."""

    scene:        LeggedV3SceneCfg       = LeggedV3SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg        = ObservationsCfg()
    actions:      CurriculumActionCfg     = CurriculumActionCfg()
    commands:     CurriculumCommandsCfg  = CurriculumCommandsCfg()
    events:       EventCfg               = EventCfg()
    rewards:      CurriculumRewardCfg    = CurriculumRewardCfg()
    terminations: TerminationsCfg        = TerminationsCfg()
    curriculum:   CurriculumCfg          = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
