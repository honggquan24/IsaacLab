# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env cfg của con lắc đơn trên xe đẩy.

Hai task dùng chung scene và action:

* ``CartPendulumEnvCfg`` — giữ thăng bằng: con lắc bắt đầu đứng sẵn, phải giữ đứng (và tự
  dựng lại nếu lỡ đổ);
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

UPRIGHT_ANGLE = 0.4
"""Ngưỡng [rad] coi là đã dựng lên — ranh giới giữa pha swing-up và pha giữ thăng bằng."""

FALL_ANGLE = 0.8
"""Ngưỡng [rad] coi là đổ hẳn, episode kết thúc. Phải LỚN HƠN ``UPRIGHT_ANGLE`` để chừa một
dải 0.4–0.8 rad cho ``swing_up`` làm tín hiệu gượng dậy trước khi bị tính là ngã."""


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
        # Số này bị chặn trên bởi CHIỀU DÀI RAY, không phải bởi mong muốn xe nhạy.
        #
        # Con lắc 0.22 m có chu kỳ 0.77 s, tức 1/4 chu kỳ là 11.5 bước ở 60 Hz — đó là quỹ
        # thời gian xe có để làm một nhịp bơm năng lượng. Xe ~0.13 kg đạp hết ga từ mép reset
        # tới đầu ray mất: 40 N → 2.6 bước, 10 N → 5.2 bước, 3 N → 9.6 bước. Trên 5 N thì xe
        # chạm đầu ray trước khi con lắc kịp nhúc nhích, và episode chết trước khi học được gì.
        #
        # Lực không phải thứ thiếu: 3 N trên quãng 0.5 m sinh 1.5 J, trong khi dựng con lắc
        # lên chỉ cần 0.024 J. Thừa 60 lần.
        scale=20.0,
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
            # 0.0 = MỌI env khởi động ở tư thế đứng, chỉ phải giữ. Con lắc vẫn đổ được giữa
            # episode và không có termination nào bắt cú đổ đó, nên reward swing_up vẫn còn
            # việc: nó là tín hiệu dựng lại sau khi ngã, chứ không còn là bài toán xuất phát.
            "hanging_prob": 0.0,
            # 0.05 rad ≈ 2.9°. Chuỗi đảo ngược có miền hút rất hẹp và hẹp dần theo số khâu:
            # 0.1 rad rơi độc lập trên từng khâu đã có thể là trạng thái không cứu được, tức
            # là ép policy học từ những ván thua sẵn.
            "angle_noise": 0.05,
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
    # (3a) pha DƯỚI: chưa dựng lên thì thưởng theo độ cao chuỗi, kéo con lắc đi lên
    swing_up = RewardTermCfg(
        func=project_mdp.swing_up_height,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"]), "upright_angle": UPRIGHT_ANGLE},
    )
    # (3b) pha TRÊN: thưởng cố định cho việc đã dựng được.
    #      Bắt buộc phải có và phải lớn hơn đỉnh của swing_up (2·cos(0.4) ≈ 1.84), nếu không
    #      vượt qua ngưỡng sẽ bị mất điểm và policy học cách lửng lơ ngay dưới ngưỡng.
    upright = RewardTermCfg(
        func=project_mdp.pendulum_is_upright,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"]), "upright_angle": UPRIGHT_ANGLE},
    )
    # (3c) pha TRÊN: L2 lệch góc như cũ, lo phần giữ cho chính xác
    pole_pos = RewardTermCfg(
        func=project_mdp.balance_pole_pos_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"]), "upright_angle": UPRIGHT_ANGLE},
    )
    # (4) định hình: giảm vận tốc xe
    cart_vel = RewardTermCfg(
        func=rewards.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"])},
    )
    # (5) định hình: giảm vận tốc góc các khâu.
    #     Ở -0.005 (giá trị của cartpole gốc) thì với chuỗi ba khâu quay 6.9 rad/s nó chỉ đóng
    #     góp -0.10/bước, quá nhẹ để cản việc quay mạnh. -0.02 cho khoảng -0.41/bước ở tốc độ
    #     đó, đủ để thừa năng lượng thành tốn kém mà vẫn không chặn nhịp bơm lúc swing-up.
    pole_vel = RewardTermCfg(
        func=rewards.joint_vel_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"])},
    )
    # (6) kéo xe về giữa ray.
    #     Trước đây term này bị bỏ vì swing-up cần quét hết ray để bơm năng lượng. Nay mọi env
    #     khởi động ở tư thế đứng nên lý do đó không còn, mà thiếu nó thì KHÔNG CÒN GÌ chống
    #     lại việc xe trôi: ``cart_out_of_rail`` là truncation nên giá trị được bootstrap, tức
    #     policy hoàn toàn BÀNG QUAN với chuyện chạm đầu ray, và bàng quan cộng trôi tự do ra
    #     đúng 100% episode chết vì chạm ray.
    #     Dạng bình phương nên gần tâm gần như bằng 0 (0.1 m → -0.005/bước, không cản việc lái
    #     bắt con lắc) mà ở mép ray thì rõ (0.5 m → -0.125/bước).
    cart_pos = RewardTermCfg(
        func=project_mdp.joint_pos_target_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"]), "wrap": False},
    )
    # (7) làm mượt lực đẩy cho đỡ giật khi quay video
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    """Kết thúc khi hết giờ, con lắc đổ, hoặc xe chạy tới đầu ray."""

    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)
    # Hai điều kiện dưới đây đều là THẤT BẠI THẬT (``time_out=False``), nên ``terminating``
    # -2.0 mới thật sự nổ. Bài đã đổi bản chất nên lập luận cũ không còn đúng:
    #
    # * Hồi còn swing-up, con lắc thõng bị phạt nặng mỗi bước nên chịu -2.0 một lần vẫn lời
    #   hơn sống tiếp — phải đánh dấu truncation để chặn động cơ tự sát. Nay mọi env khởi
    #   động ở tư thế đứng, phần thưởng thường trực là dương (alive +1.0, upright +3.0), nên
    #   kết thúc sớm đã tự nó là mất mát.
    # * Ngược lại, truncation lại sinh ra đúng lỗi ngược: giá trị được bootstrap nên policy
    #   BÀNG QUAN với chuyện chạm đầu ray, và ở vòng 146 của con lắc ba thì 100% episode chết
    #   vì chạm ray sau vỏn vẹn 39 bước (0.65 s).
    #
    # Quan trọng: HAI term phải cùng loại. Nếu chỉ đổ mới bị phạt còn chạm ray thì miễn phí,
    # policy sẽ học cách phóng vào đầu ray TRƯỚC KHI con lắc kịp đổ để né -2.0.
    pendulum_fell = TerminationTermCfg(
        func=project_mdp.pendulum_fell,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"]), "limit_angle": FALL_ANGLE},
    )
    # Không có term này thì sau cú đổ episode còn lê thêm ~19.8 s vung vẩy vô nghĩa: 1% dữ liệu
    # là thăng bằng thật, 99% là rác. Cắt ngay lúc đổ biến bài toán thành đúng một câu hỏi
    # "trụ được bao lâu", và đó là tín hiệu mà PPO gán công trạng tốt nhất.
    cart_out_of_rail = TerminationTermCfg(
        func=project_mdp.cart_out_of_rail,
        time_out=False,
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
    """Thêm phần bám mốc. Cũng chỉ chạy ở pha TRÊN, xem ``joint_pos_command_l2``."""

    #     Trần an toàn của trọng số này là 1.65, đừng vượt. Term bị cổng "đã dựng" chặn, mà nó
    #     là PHẠT, nên buông con lắc xuống là tắt luôn phần phạt: nặng quá thì buông lại lời
    #     hơn giữ. Ở sai số tệ nhất (0.84 m): w=1.0 giữ được +2.30 so với +1.84 nếu buông,
    #     còn w=2.0 chỉ còn +1.60 — policy sẽ học cách thả con lắc cho khỏi bị phạt bám mốc.
    track_position = RewardTermCfg(
        func=project_mdp.joint_pos_command_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"]),
            "pole_cfg": SceneEntityCfg("robot", joint_names=["Revolute_.*"]),
            "upright_angle": UPRIGHT_ANGLE,
        },
    )


@configclass
class CartPendulumPositionEnvCfg(CartPendulumEnvCfg):
    """Xe đẩy bám vị trí mục tiêu trong khi giữ con lắc thăng bằng."""

    commands: CommandsCfg = CommandsCfg()
    observations: PositionObservationsCfg = PositionObservationsCfg()
    rewards: PositionRewardCfg = PositionRewardCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # mốc do command sinh ra, không phải giữa ray: để nguyên cart_pos là hai term kéo xe
        # về hai chỗ khác nhau, mốc càng xa tâm thì càng bị chính cart_pos ghì lại
        self.rewards.cart_pos.weight = 0.0


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
