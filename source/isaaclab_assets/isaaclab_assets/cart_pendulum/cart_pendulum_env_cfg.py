# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env cfg của con lắc đơn trên xe đẩy.

Hai task dùng chung scene và action:

* ``CartPendulumEnvCfg`` — swing-up: con lắc bắt đầu thõng xuống, phải lắc lên rồi giữ đứng;
* ``CartPendulumPositionEnvCfg`` — vừa giữ con lắc đứng vừa chạy tới mốc vị trí được lệnh.

Toạ độ đọc từ USD: ray nằm dọc trục **Y**, giới hạn ±0.555 m; ``Revolute_1`` bằng 0 là con
lắc thõng xuống, tư thế đứng là π và đã được đặt làm vị trí khớp mặc định trong
``CART_PENDULUM_CFG``. Mọi reward/termination đo lệch so với mặc định đó.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import actions, events, rewards, terminations
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp as project_mdp
from .cart_pendulum_cfg import CART_PENDULUM_CFG, CART_PENDULUM_RAIL_LIMIT

RAIL_AXIS = (0.0, 1.0, 0.0)
"""Hướng ray trong world. Đọc từ USD: rack trải từ y=-0.555 tới y=+0.555."""


@configclass
class CartPendulumSceneCfg(InteractiveSceneCfg):
    """Scene: một mặt sàn, một đèn dome, một robot."""

    num_envs: int = 1

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: Articulation = CART_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    """Chỉ khớp trượt được điều khiển; khớp con lắc thụ động nên không nằm trong action."""

    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Slider_1"],
        # Gia tốc mới là thứ quyết định xe "nhạy" hay không, chứ không phải tốc độ đỉnh.
        # Xe con lắc đơn ~0.13 kg → 40 N là ~300 m/s²; con lắc ba nặng gấp đôi nên còn
        # ~150 m/s². Đây là số cần chỉnh trước tiên nếu thấy xe phản ứng chậm.
        scale=40.0,
    )


@configclass
class ObservationsCfg:
    """Quan sát: vị trí và vận tốc khớp, lấy tương đối so với mặc định.

    Dùng bản ``_rel`` để góc con lắc đưa vào mạng là lệch so với tư thế đứng (quanh 0) chứ
    không phải giá trị thô quanh π.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Mỗi lần reset thì xê dịch xe và con lắc quanh tư thế mặc định."""

    reset_cart = EventTerm(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_pendulum = EventTerm(
        func=project_mdp.reset_pendulum_chain,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
            # bắt đầu ở tư thế thõng xuống: đây là bài swing-up, phải lắc từ dưới lên
            "hanging": True,
            "angle_noise": 0.1,
            "velocity_noise": 0.05,
        },
    )


@configclass
class RewardCfg:
    """Reward theo đúng dạng cartpole gốc của Isaac Lab: L2 cho vị trí, L1 cho vận tốc.

    Không dùng exp. L2 có độ dốc ở mọi góc nên swing-up vẫn có cái để bám, còn
    ``exp(-e²/std²)`` thì ở tư thế thõng đã bão hoà về 0 và phẳng lì.
    """

    # (1) thưởng đều mỗi bước còn sống
    alive = RewardTermCfg(func=rewards.is_alive, weight=1.0)
    # (2) phạt khi kết thúc vì thất bại
    terminating = RewardTermCfg(func=rewards.is_terminated, weight=-2.0)
    # (3) việc chính: đưa cả chuỗi về tư thế đứng
    pole_pos = RewardTermCfg(
        func=project_mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"])},
    )
    # (4) định hình: xe đừng trôi ra đầu ray
    cart_pos = RewardTermCfg(
        func=project_mdp.joint_pos_target_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"]), "wrap": False},
    )
    # (5) định hình: giảm vận tốc xe
    cart_vel = RewardTermCfg(
        func=rewards.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"])},
    )
    # (6) định hình: giảm vận tốc góc các khâu
    pole_vel = RewardTermCfg(
        func=rewards.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"])},
    )
    # (7) làm mượt lực đẩy cho đỡ giật khi quay video
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    """Kết thúc khi hết giờ, con lắc đổ, hoặc xe chạy tới đầu ray."""

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)
    # KHÔNG kết thúc khi con lắc đổ: bài này bắt đầu từ tư thế thõng, đổ là trạng thái xuất phát.
    #
    # Chạm đầu ray đánh dấu time_out=True (cắt ngang) chứ không phải thất bại, và đây là chỗ dễ
    # sai: với reward L2, con lắc thõng bị phạt tới -π² ≈ -9.9 mỗi bước, nên nếu kết thúc sớm
    # được tính là thất bại thì chịu -2.0 một lần vẫn lời hơn hẳn việc sống tiếp — policy sẽ học
    # cách lao vào đầu ray cho xong. Đánh dấu cắt ngang thì value được bootstrap, hết động cơ đó.
    cart_out_of_rail = TerminationTermCfg(
        func=project_mdp.cart_out_of_rail,
        time_out=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"]),
            "limit": CART_PENDULUM_RAIL_LIMIT - 0.05,
        },
    )


@configclass
class CartPendulumEnvCfg(ManagerBasedRLEnvCfg):
    """Giữ con lắc thăng bằng trên xe đẩy."""

    scene: CartPendulumSceneCfg = CartPendulumSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # Tần số điều khiển 60 Hz. Ở 30 Hz cũ, với trần 20 m/s thì mỗi bước điều khiển xe đi
        # 0.67 m — hơn 60% chiều dài ray, tức policy không kịp lái: nới trần tốc độ mà không
        # nới nhịp ra quyết định thì chỉ đổi lấy việc xe đâm đầu ray.
        self.decimation = 1  # tần số điều khiển = 60/1 = 60 Hz
        self.episode_length_s = 20.0

        # ray nằm dọc Y nên đặt camera trên trục X, nếu không sẽ nhìn dọc thân ray
        self.viewer.eye = (2.5, 0.0, 0.9)
        self.viewer.lookat = (0.0, 0.0, 0.3)

        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation


##
# Biến thể bám vị trí: xe vừa giữ con lắc đứng vừa chạy tới mốc vị trí được lệnh.
##


@configclass
class CommandsCfg:
    """Lệnh vị trí cho xe đẩy."""

    cart_position = project_mdp.CartPositionCommandCfg(
        joint_name="Slider_1",
        # đổi mốc sau mỗi 3–5 s: đủ lâu để xe tới nơi và đứng yên một nhịp trước khi có mốc mới
        resampling_time_range=(3.0, 5.0),
        # chỉ dùng 60% chiều dài ray, chừa biên để xe còn chỗ giảm tốc
        limit_ratio=0.6,
        rail_axis=RAIL_AXIS,
        # hiện quả cầu đỏ đánh dấu mốc — quay video nhìn ra ngay xe đang bám cái gì
        debug_vis=True,
    )


@configclass
class PositionObservationsCfg(ObservationsCfg):
    """Thêm lệnh vào cuối vector quan sát (4 → 5 chiều)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        cart_position_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "cart_position"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class PositionRewardCfg(RewardCfg):
    """Thêm phần bám mốc; phần kéo xe về giữa ray bị tắt trong ``__post_init__`` của env."""

    track_position = RewardTermCfg(
        func=project_mdp.joint_pos_command_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"])},
    )


@configclass
class CartPendulumPositionEnvCfg(CartPendulumEnvCfg):
    """Xe đẩy bám vị trí mục tiêu trong khi giữ con lắc thăng bằng."""

    commands: CommandsCfg = CommandsCfg()
    observations: PositionObservationsCfg = PositionObservationsCfg()
    rewards: PositionRewardCfg = PositionRewardCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # mốc mới là thứ quyết định xe đứng ở đâu, giữ thêm lực kéo về giữa ray là mâu thuẫn
        self.rewards.cart_pos = None


@configclass
class CartPendulumPositionPlayEnvCfg(CartPendulumPositionEnvCfg):
    """Cấu hình dùng lúc quay video: ít env, episode dài để clip không bị reset giữa chừng."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        # 60 s liền mạch, khớp với --video_length 3600 ở 60 Hz
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
