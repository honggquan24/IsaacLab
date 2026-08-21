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
from isaaclab.utils import configclass

from .cart_pendulum_cfg import CART_PENDULUM_CFG
from .mdp.commands import CartPositionCommandCfg
from .mdp.rewards import *  # noqa: F403


@configclass
class CartpoleRobotV1SceneConfig(InteractiveSceneCfg):
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
    robot: Articulation = CART_PENDULUM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


@configclass
class ActionsCfg:
    # joint_effort = mdp.actions.actions_cfg.JointEffortActionCfg
    # (joint_names=["Slider_1"],asset_name="robot",scale=1.0)
    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Slider_1", "Revolute_1"],
        scale={
            "Slider_1": 100.0,
            "Revolute_1": 0.0,
        },
        debug_vis=True,
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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # on reset
    reset_cart_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class RewardCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward - encourage survival
    alive = RewardTermCfg(func=rewards.is_alive, weight=1.0)

    # (2) Failure penalty - penalize termination0
    terminating = RewardTermCfg(func=rewards.is_terminated, weight=-2.0)
    rw_joint_pos = RewardTermCfg(func=cartpole_reward_joint_pos, weight=2.0)

    rw_joint_vel = RewardTermCfg(func=cartpole_reward_joint_vel, weight=-3)
    rw_fall = RewardTermCfg(func=cartpole_reward_fall, weight=-2.0)


@configclass
class TerminationsCfg:
    """Termination configuration for the cart pendulum environment."""

    # TIME OUT - Episode ends after episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,  # Mark as timeout (not failure)
    )


@configclass
class CartPendulumEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""

    # Scene settings
    scene: CartpoleRobotV1SceneConfig = CartpoleRobotV1SceneConfig(
        num_envs=1,
        env_spacing=2.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2  # Control freq = 60/2 = 30 Hz
        self.episode_length_s = 100  # Episode duration

        # Viewer settings
        self.viewer.eye = (0.0, 5.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point

        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps


##
# Biến thể bám vị trí: xe vừa giữ con lắc đứng vừa chạy tới mốc vị trí được lệnh.
##


@configclass
class CommandsCfg:
    """Lệnh vị trí cho xe đẩy."""

    cart_position = CartPositionCommandCfg(
        joint_name="Slider_1",
        # đổi mốc sau mỗi 3–5 s: đủ lâu để xe đi tới nơi và đứng yên một nhịp trước khi có mốc mới
        resampling_time_range=(3.0, 5.0),
        # chỉ dùng 60% chiều dài ray, chừa biên để xe còn chỗ giảm tốc
        limit_ratio=0.6,
        # hiện quả cầu đỏ đánh dấu mốc — quay video nhìn ra ngay xe đang bám cái gì
        debug_vis=True,
    )


@configclass
class PositionObservationsCfg(ObservationsCfg):
    """Quan sát của task bám vị trí: thêm lệnh vào cuối vector quan sát (4 → 5 chiều)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        cart_position_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "cart_position"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class PositionRewardCfg:
    """Reward của task bám vị trí.

    Viết riêng chứ không kế thừa :class:`RewardCfg`: các hàm ``cartpole_reward_*`` gộp cả vị trí
    lẫn vận tốc con lắc vào một sai số nên không tách được phần bám vị trí xe ra.
    """

    # (1) sống sót / ngã
    alive = RewardTermCfg(func=rewards.is_alive, weight=1.0)
    terminating = RewardTermCfg(func=rewards.is_terminated, weight=-4.0)

    # (2) giữ con lắc dựng đứng
    upright = RewardTermCfg(func=upright_pendulum_exp, weight=2.0, params={"std": 0.35})
    pendulum_rate = RewardTermCfg(func=pendulum_ang_vel_l2, weight=-0.02)

    # (3) bám mốc vị trí và dừng hẳn tại đó
    track_position = RewardTermCfg(func=track_cart_position_exp, weight=3.0, params={"std": 0.25})
    stop_at_goal = RewardTermCfg(func=cart_velocity_near_goal_l2, weight=-0.2, params={"std": 0.25})

    # (4) làm mượt lực đẩy cho đỡ giật khi quay video
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.005)


@configclass
class PositionTerminationsCfg(TerminationsCfg):
    """Kết thúc episode khi con lắc đổ, để khỏi phí bước học ở tư thế không cứu được."""

    pendulum_fell = TerminationTermCfg(
        func=terminations.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1"]),
            "bounds": (-0.8, 0.8),
        },
    )


@configclass
class CartPendulumPositionEnvCfg(CartPendulumEnvCfg):
    """Xe đẩy bám vị trí mục tiêu trong khi giữ con lắc thăng bằng."""

    commands: CommandsCfg = CommandsCfg()
    observations: PositionObservationsCfg = PositionObservationsCfg()
    rewards: PositionRewardCfg = PositionRewardCfg()
    terminations: PositionTerminationsCfg = PositionTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # episode ngắn hơn task cân bằng: mỗi lượt vẫn kịp 4–6 mốc mà reset dày hơn
        self.episode_length_s = 20.0


@configclass
class CartPendulumPositionPlayEnvCfg(CartPendulumPositionEnvCfg):
    """Cấu hình dùng lúc quay video: ít env, episode dài để clip không bị reset giữa chừng."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        # 60 s liền mạch, khớp với --video_length 1800 ở 30 Hz
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
