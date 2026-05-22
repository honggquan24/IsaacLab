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

from isaaclab.managers import CurriculumTermCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import commands

from ..legged_v3_wheel_env_cfg import (
    LeggedV3SceneCfg,
    ActionCfg,
    ObservationsCfg,
    EventCfg,
    RewardCfg,
    TerminationsCfg,
)
from .mdp import expand_velocity_command_range


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

    scene:        LeggedV3SceneCfg      = LeggedV3SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg       = ObservationsCfg()
    actions:      ActionCfg             = ActionCfg()
    commands:     CurriculumCommandsCfg = CurriculumCommandsCfg()
    events:       EventCfg              = EventCfg()
    rewards:      RewardCfg             = RewardCfg()
    terminations: TerminationsCfg       = TerminationsCfg()
    curriculum:   CurriculumCfg         = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
