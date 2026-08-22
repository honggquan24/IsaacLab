# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tầng navigation học TỪ ĐẦU — vừa giữ thăng bằng vừa chạy tới đích, một mạng duy nhất.

Khác :mod:`.navigation_pretrained_env_cfg` (cascade hai tầng): ở đây policy xuất thẳng mô-men
bánh và phải tự học cả hai việc cùng lúc. Bài này **khó hơn hẳn** — không có tầng nào lo phần
cân bằng — nên nó dùng để đối chứng cho thấy vì sao chia tầng đáng giá, chứ không phải phương
án chính cho video.

Toàn bộ file là **một biến thể của** :class:`~..balance_env_cfg.BalanceCarEnvCfg`: cùng scene,
cùng action, cùng event, cùng termination. Chỉ đổi ba thứ — lệnh, quan sát, reward. Bản trước
chép lại toàn bộ scene, IMU, ground, action và observation thành 357 dòng riêng; hai bản chép
sau đó trôi khỏi nhau và ma sát/tần số/ngưỡng ngã ở hai nơi không còn giống nhau.

.. note::
    Không có term thưởng **hướng cuối** ở đây. ``UniformPose2dCommand`` tính sai số hướng theo
    ``data.heading_w``, tức góc của **body +X** trong world; xe này tiến theo **body +Y** nên
    sai số đó luôn lệch một hằng số 90°. Đích chỉ có vị trí, không có hướng.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..balance_env_cfg import BalanceCarEnvCfg
from .mdp.rewards import position_command_error_tanh


@configclass
class NavigationCommandsCfg:
    """Một điểm đích ngẫu nhiên trong ô env."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(15.0, 15.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            # bắt buộc phải khai, nhưng không term reward nào đọc tới — xem note ở đầu module
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class NavigationObservationsCfg:
    """Bằng quan sát của tầng thấp, thay ``velocity_commands`` bằng vị trí đích."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class NavigationRewardsCfg:
    """Trọng số là **điểm mỗi giây** (Isaac Lab nhân reward với ``step_dt``)."""

    # -- tới đích: hai thang, thô để kéo từ xa và mịn để đứng đúng chỗ. Đây là cặp mà mẫu
    #    navigation của Isaac Lab dùng.
    position_tracking = RewTerm(
        func=position_command_error_tanh, weight=2.0, params={"command_name": "pose_command", "std": 2.0}
    )
    position_tracking_fine = RewTerm(
        func=position_command_error_tanh, weight=2.0, params={"command_name": "pose_command", "std": 0.2}
    )

    # -- giữ thăng bằng: cùng bộ term với tầng thấp, cùng trọng số
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # -- lực bánh mượt
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class BalanceCarNavigationEnvCfg(BalanceCarEnvCfg):
    """Giữ thăng bằng + chạy tới đích, học từ đầu bằng một mạng."""

    commands: NavigationCommandsCfg = NavigationCommandsCfg()
    observations: NavigationObservationsCfg = NavigationObservationsCfg()
    rewards: NavigationRewardsCfg = NavigationRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # một episode = trọn một lần bốc đích
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]


@configclass
class BalanceCarNavigationEnvCfg_PLAY(BalanceCarNavigationEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
