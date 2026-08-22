# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation environment using pre-trained balance policy for the balance car.

This configuration uses a pre-trained balance policy as low-level controller
and trains a high-level navigation policy on top of it.
"""

# ruff: noqa: F405  (env cfg dùng tên đến từ `import *` của package mdp trong cùng dự án)

import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..balance_env_cfg import BalanceCarEnvCfg
from ..mdp.observations import angl_vel_b, lin_vel_b, obs_body_pitch, obs_body_roll, obs_body_yaw
from ..mdp.rewards import cover_flat_exp, cover_flat_l2
from ..mdp.terminations import reset_when_fall
from .mdp.commands import PathCommandCfg
from .mdp.pre_trained_policy_action import PreTrainedBalancePolicyActionCfg, latest_exported_policy
from .mdp.rewards import *  # noqa: F403

# Load low-level balance environment config
LOW_LEVEL_ENV_CFG = BalanceCarEnvCfg()


@configclass
class LowLevelObservationsCfg(ObsGroup):
    """Observations for the low-level balance policy.

    This must match the observation space that the balance policy was trained on.
    Based on balance_env_cfg.py ObservationsCfg.PolicyCfg (without obs_pos_w).
    """

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
    # Chỗ giữ sẵn cho lệnh vận tốc; PreTrainedBalancePolicyAction ghi đè func lúc khởi tạo để
    # nó trả về action của tầng cao. Thứ tự term phải khớp đúng PolicyCfg của tầng thấp.
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ActionsCfg:
    """Action configuration using pre-trained balance policy."""

    pre_trained_policy_action: PreTrainedBalancePolicyActionCfg = PreTrainedBalancePolicyActionCfg(
        asset_name="robot",
        # Tự lấy run mới nhất của tầng thấp. Train Isaac-Balance-Car rồi chạy play.py một lần
        # để nó export ra logs/rsl_rl/carbalance_ppo/<run>/exported/policy.pt là dùng được ngay,
        # không phải quay lại sửa file này.
        policy_path=latest_exported_policy("carbalance_ppo"),
        # PHẢI bằng decimation của env tầng thấp (2 → 30 Hz). apply_actions() được gọi mỗi
        # bước vật lý 60 Hz, nên để 1 là policy thăng bằng bị hỏi ở 60 Hz trong khi nó được
        # train ở 30 Hz — sai tần số thì vận tốc/gia tốc nó thấy lệch hẳn so với lúc học.
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_effort,
        low_level_observations=LowLevelObservationsCfg(),
        # debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for navigation policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """High-level observations for navigation."""

        # Robot state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Lệnh quỹ đạo: [lệch dọc, lệch ngang, lệch hướng, tốc độ mục tiêu]
        path_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "path_command"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for navigation."""

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
class CommandsCfg:
    """Command configuration for navigation."""

    # BÁM QUỸ ĐẠO thay cho chạy tới một điểm đích. Mục tiêu chạy liên tục trên đường cong kín,
    # nên không có khái niệm "đã tới nơi" — xem mdp/commands.py.
    path_command = PathCommandCfg(
        asset_name="robot",
        path_types=("circle", "figure8"),
        radius_range=(1.0, 2.0),
        # phải nằm trong dải lệnh của tầng thấp (lin_vel_x = ±0.5 m/s), nếu không mục tiêu
        # chạy nhanh hơn khả năng bám và tín hiệu học chỉ còn là "luôn tụt lại"
        speed_range=(0.15, 0.35),
        # một quỹ đạo cho trọn một episode: đổi đường giữa chừng thì phần lớn thời gian là
        # chạy tới đường mới chứ không phải bám đường
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
    )


@configclass
class RewardsCfg:
    """Bám quỹ đạo, giữ thăng bằng, chạy mượt.

    Bộ reward "chạy tới đích" cũ đã bỏ hết vì nó giải bài khác:

    * ``goal_progress`` thưởng theo mức giảm khoảng cách — vô nghĩa khi đích tự chạy. Nó còn
      giữ trạng thái trong ``env.extras["prev_dist"]``, một dict DÙNG CHUNG cho mọi env và
      không được dọn lúc reset, nên ngay sau mỗi lần reset nó cho một cú thưởng/phạt rác;
    * ``reached_bonus`` thưởng khi vào bán kính 0.3 m — mục tiêu không đứng yên nên "tới nơi"
      không tồn tại;
    * ``velocity_to_goal``, ``heading_alignment`` cũng đều gắn với một đích đứng yên.
    """

    # =====================================================
    # Bám quỹ đạo — phần chính
    # =====================================================
    path_position = RewTerm(
        func=path_position_exp,
        weight=6.0,
        params={"command_name": "path_command", "std": 0.5},
    )
    # bám vị trí thôi thì xe vẫn có thể ĐI LÙI hoặc trượt ngang qua khúc cua mà vẫn ăn điểm.
    # Term này bắt mũi xe quay đúng chiều tiếp tuyến — nó quyết định video có ra hồn không.
    path_heading = RewTerm(
        func=path_heading_exp,
        weight=2.0,
        params={"command_name": "path_command", "std": 0.6},
    )
    # tách riêng phần lệch NGANG: tụt lại sau vài chục phân là bình thường và tự sửa được,
    # còn cắt cua ra ngoài đường mới đúng nghĩa đi sai quỹ đạo
    path_lateral = RewTerm(
        func=path_lateral_l2,
        weight=-2.0,
        params={"command_name": "path_command"},
    )

    # =====================================================
    # Giữ thăng bằng — cùng hàm với tầng thấp
    # =====================================================
    cover_flat = RewTerm(func=cover_flat_l2, weight=-5.0)
    cover_flat_bonus = RewTerm(func=cover_flat_exp, weight=2.0, params={"std": 0.05})

    # =====================================================
    # Kết thúc sớm
    # =====================================================
    # -300 của bản cũ là quá tay: nó áp đảo mọi tín hiệu bám đường, biến bài toán thành
    # "đừng ngã" và policy học cách đứng yên tại chỗ cho an toàn. Đặt ngang tầm với phần
    # thưởng bám đường tích luỹ trong vài giây.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # =====================================================
    # Lệnh xuất ra tầng thấp phải mượt
    # =====================================================
    # Tầng cao chạy 6 Hz. Lệnh vận tốc nhảy loạn mỗi bước thì tầng thấp — vốn được train trên
    # lệnh đổi mỗi 3-6 s — gặp phân phối hoàn toàn khác lúc học và bám rất tệ.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination configuration for navigation."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=reset_when_fall)


@configclass
class BalanceCarNavigationPretrainedEnvCfg(ManagerBasedRLEnvCfg):
    """Navigation environment using pre-trained balance policy."""

    # Use the same scene as balance task
    scene = LOW_LEVEL_ENV_CFG.scene

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # Use same simulation settings as low-level env
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation

        # Higher decimation for navigation (low-level runs faster)
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 5

        # Một episode = trọn một lần bốc quỹ đạo. Ở 6 Hz thì 20 s = 120 bước tầng cao, đủ để
        # chạy hết ~1 vòng đường bán kính 1.5 m ở 0.3 m/s.
        self.episode_length_s = self.commands.path_command.resampling_time_range[1]

        # Viewer settings
        self.viewer.eye = (0.0, 8.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # Scene settings
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0


@configclass
class BalanceCarNavigationPretrainedEnvCfg_PLAY(BalanceCarNavigationPretrainedEnvCfg):
    """Play configuration for navigation with pre-trained policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
