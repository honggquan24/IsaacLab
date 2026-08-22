# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: F405  (env cfg dùng tên đến từ `import *` của package mdp trong cùng dự án)
import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import actions, rewards, terminations
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

# from .cartpole_v2_cfg import CARTPOLE_V2_CFG
from .balance_car_cfg import BALANCE_CAR_CFG
from .mdp.observations import *  # noqa: F403
from .mdp.rewards import *  # noqa: F403
from .mdp.terminations import *  # noqa: F403


@configclass
class BalanceCarSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the legged robot environment."""

    num_envs: int = 1

    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    cfg_ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Add robot
    robot: Articulation = BALANCE_CAR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    # Add robot
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/balance_robot/balance_robot/balance_body",
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.2),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        update_period=0.0,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    # joint_effort = mdp.actions.actions_cfg.JointEffortActionCfg
    # (joint_names=["Slider_1"],asset_name="robot",scale=1.0)
    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Revolute_1", "Revolute_2"],
        scale={
            "Revolute_1": 100.0,
            "Revolute_2": 100.0,
        },
        debug_vis=True,
    )


@configclass
class CommandsCfg:
    """Lệnh vận tốc cho tầng thấp.

    Đây là điểm khác cốt lõi so với bản cũ: tầng thấp không còn học "đứng yên giữ thăng bằng"
    mà học "vừa giữ thăng bằng vừa chạy theo lệnh". Nhờ vậy tầng navigation ở trên chỉ cần
    xuất ra lệnh vận tốc là điều khiển được, không phải cộng thẳng vào mô-men bánh.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        # 20% số env được lệnh đứng yên — giữ nguyên bài "cân bằng tại chỗ" trong phân phối
        rel_standing_envs=0.2,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            # xe hai bánh vi sai không đi ngang được
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        pitch_angl_p = ObsTerm(
            func=obs_body_pitch,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        pitch_angl_r = ObsTerm(
            func=obs_body_roll,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        pitch_angl_y = ObsTerm(
            func=obs_body_yaw,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        l_vel = ObsTerm(
            func=lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        a_vel = ObsTerm(
            func=angl_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        # Bỏ obs_pos_world: vị trí tuyệt đối trong env làm policy gắn với một chỗ cụ thể,
        # vô dụng khi tầng navigation muốn nó chạy đi bất kỳ đâu.
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        pitch_angl_p = ObsTerm(
            func=obs_body_pitch,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        pitch_angl_r = ObsTerm(
            func=obs_body_roll,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        pitch_angl_y = ObsTerm(
            func=obs_body_yaw,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        l_vel = ObsTerm(
            func=lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        a_vel = ObsTerm(
            func=angl_vel_b,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # on reset
    reset_pole_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_[1-2]"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class RewardCfg:
    """Giữ thăng bằng VÀ bám lệnh vận tốc.

    Ba term của bản cũ đã bị bỏ vì chúng thưởng cho việc **đứng yên**, tức là chống lại lệnh:

    * ``reward_vel`` thưởng vận tốc bánh về 0;
    * ``reward_li_vel`` thưởng vận tốc tiến về 0;
    * ``reward_angle_y`` thưởng góc yaw bám 90° — một hướng TUYỆT ĐỐI trong world, giữ nó thì
      xe không rẽ được.
    """

    alive = RewardTermCfg(func=rewards.is_alive, weight=1.0)
    terminating = RewardTermCfg(func=rewards.is_terminated, weight=-2.0)

    # giữ thân dựng (tư thế đứng của xe này là |roll| = 90°)
    reward_angl = RewardTermCfg(func=reward_angle_r, weight=3.5)
    reward_bonus_when_up = RewardTermCfg(func=bonus_reward, weight=2.0)
    reward_r_rate = RewardTermCfg(func=reward_roll_rate, weight=0.4)

    # bám lệnh
    track_lin_vel = RewardTermCfg(func=track_lin_vel_exp, weight=2.0, params={"std": 0.25})
    track_ang_vel = RewardTermCfg(func=track_ang_vel_exp, weight=1.0, params={"std": 0.5})


@configclass
class TerminationsCfg:
    # TIME OUT - Episode ends after episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,  # Mark as timeout (not failure)
    )
    when_fall = TerminationTermCfg(
        func=reset_when_fall,
    )


@configclass
class BalanceCarEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""

    # Scene settings
    scene: BalanceCarSceneCfg = BalanceCarSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2  # Control freq = 60/2 = 30 Hz
        self.episode_length_s = 20.0  # đủ cho 4-6 lệnh mỗi episode

        # Viewer settings
        self.viewer.eye = (0.0, 5.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point

        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps
