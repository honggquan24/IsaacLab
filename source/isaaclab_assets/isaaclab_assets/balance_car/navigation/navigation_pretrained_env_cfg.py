# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tầng navigation chạy trên policy thăng bằng đã train — bám quỹ đạo.

Dựng theo ``isaaclab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py``,
mẫu cascade chuẩn của Isaac Lab. Khác mẫu gốc ở một chỗ: mục tiêu không phải một điểm đứng yên
mà là **một điểm chạy liên tục trên đường cong kín** (xem :mod:`.mdp.commands`).

Kiến trúc
---------
::

    tầng cao 10 Hz  ──(vx, vy, wz)──▶  policy thăng bằng 50 Hz  ──mô-men──▶  bánh xe

Tầng cao chỉ xuất lệnh vận tốc; nó **không đụng tới mô-men bánh**. Policy tầng thấp đã được
train để bám đúng loại lệnh này nên nó tự phối hợp nghiêng thân với quay bánh.

Điều đã sửa so với bản trước
----------------------------
Bản trước **chép tay** danh sách quan sát của tầng thấp vào một class
``LowLevelObservationsCfg`` riêng, kèm chú thích "thứ tự term phải khớp đúng PolicyCfg của tầng
thấp". Đó là một ràng buộc không ai kiểm được: sửa quan sát ở :mod:`..balance_env_cfg` mà quên
sửa ở đây thì policy nhận vào một vector trộn sai thứ tự, chạy trơn tru và cho kết quả vô
nghĩa. Ở đây dùng thẳng ``LOW_LEVEL_ENV_CFG.observations.policy`` — đúng cách mẫu gốc làm,
và không còn gì để lệch.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..balance_env_cfg import BALANCE_CAR_FALL_ANGLE, BalanceCarEnvCfg
from ..balance_env_cfg import EventCfg as LowLevelEventCfg
from .mdp.commands import PathCommandCfg
from .mdp.pre_trained_policy_action import PreTrainedBalancePolicyActionCfg, latest_exported_policy
from .mdp.rewards import path_heading_exp, path_lateral_l2, path_position_exp

LOW_LEVEL_ENV_CFG = BalanceCarEnvCfg()

# Policy tầng thấp được huấn luyện KHÔNG có nhiễu quan sát ở đây: nó đã đóng băng, thêm nhiễu
# vào đầu vào của nó chỉ làm nó tệ đi chứ không dạy được gì cho ai.
LOW_LEVEL_ENV_CFG.observations.policy.enable_corruption = False


@configclass
class ActionsCfg:
    """Action của tầng cao = lệnh vận tốc cho tầng thấp."""

    pre_trained_policy_action: PreTrainedBalancePolicyActionCfg = PreTrainedBalancePolicyActionCfg(
        asset_name="robot",
        # Tự lấy run mới nhất. Train Isaac-Balance-Car rồi chạy play.py một lần để nó export
        # logs/rsl_rl/carbalance_ppo/<run>/exported/policy.pt là dùng được ngay.
        policy_path=latest_exported_policy("carbalance_ppo"),
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_effort,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Quan sát của tầng cao — đúng bộ của mẫu navigation, thêm vận tốc góc.

    Thêm ``base_ang_vel`` vì xe hai bánh vi sai rẽ bằng chênh lệch tốc độ bánh: không nhìn
    thấy mình đang quay nhanh cỡ nào thì không điều tiết được lệnh ``wz``.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        path_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "path_command"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg(LowLevelEventCfg):
    """Kế thừa nguyên phần random miền của tầng thấp, chỉ nhẹ tay hơn lúc reset.

    Giữ nguyên ``physics_material`` / ``add_frame_mass`` / ``frame_com``: tầng cao phải chịu
    được đúng dải robot mà tầng thấp đã học, nếu không nó sẽ học một quan hệ "lệnh → chuyển
    động" chỉ đúng cho một con xe.
    """

    def __post_init__(self) -> None:
        # Tầng cao đang học bám đường, không học gượng dậy. Để nhiễu nghiêng lớn như tầng thấp
        # thì phần đầu mỗi episode là tầng thấp đang cứu xe, tầng cao không điều khiển được gì
        # mà vẫn bị chấm điểm cho quãng đó.
        self.reset_base.params["pose_range"] = {"yaw": (-math.pi, math.pi), "roll": (-0.05, 0.05)}
        self.reset_base.params["velocity_range"] = {}


@configclass
class CommandsCfg:
    """Quỹ đạo để bám. Xem :mod:`.mdp.commands`."""

    path_command = PathCommandCfg(
        asset_name="robot",
        path_types=("circle", "figure8"),
        radius_range=(1.0, 2.0),
        speed_range=(0.4, 0.9),
        # một quỹ đạo cho trọn một episode: đổi đường giữa chừng thì phần lớn thời gian là
        # chạy tới đường mới chứ không phải bám đường
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
    )


@configclass
class RewardsCfg:
    """Trọng số là **điểm mỗi giây** (Isaac Lab nhân reward với ``step_dt``)."""

    # -- bám quỹ đạo
    path_position = RewTerm(func=path_position_exp, weight=6.0, params={"command_name": "path_command", "std": 0.5})
    path_heading = RewTerm(func=path_heading_exp, weight=2.0, params={"command_name": "path_command", "std": 0.6})
    path_lateral = RewTerm(func=path_lateral_l2, weight=-2.0, params={"command_name": "path_command"})

    # -- giữ nắp phẳng, cùng term với tầng thấp
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # -- lệnh xuất ra tầng thấp phải mượt
    # Tầng cao chạy 10 Hz. Lệnh nhảy loạn mỗi bước thì tầng thấp — vốn được train trên lệnh đổi
    # mỗi 5 s — gặp phân phối hoàn toàn khác lúc học và bám rất tệ.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # -- ngã
    # -300 của bản cũ áp đảo mọi tín hiệu bám đường và biến bài toán thành "đừng ngã", policy
    # học cách đứng yên cho an toàn. Nhân với dt → -100 × 0.1 = -10 một lần ở 10 Hz, xấp xỉ
    # 1.5 giây bám đường hoàn hảo.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_fell = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": BALANCE_CAR_FALL_ANGLE})


@configclass
class BalanceCarNavigationPretrainedEnvCfg(ManagerBasedRLEnvCfg):
    """Bám quỹ đạo, tầng thấp là policy thăng bằng đã train."""

    scene = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.sim.physics_material = LOW_LEVEL_ENV_CFG.sim.physics_material
        # tầng cao 10 Hz = 200 / (4 × 5). Mẫu anymal dùng ×10 (5 Hz); ở đây ×5 vì mục tiêu
        # chạy liên tục chứ không đứng yên, lệnh cập nhật thưa quá thì xe cắt cua.
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 5
        # một episode = trọn một lần bốc quỹ đạo
        self.episode_length_s = self.commands.path_command.resampling_time_range[1]

        self.viewer.eye = (0.0, 8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class BalanceCarNavigationPretrainedEnvCfg_PLAY(BalanceCarNavigationPretrainedEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
