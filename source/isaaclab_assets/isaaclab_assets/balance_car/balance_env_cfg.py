# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Xe hai bánh tự cân bằng — bám lệnh vận tốc.

Dựng theo ``isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py``, mẫu chuẩn
của Isaac Lab cho bài "bám lệnh ``(vx, vy, wz)`` với thân nổi". Xe cân bằng **đúng là bài đó**,
chỉ khác ở chỗ robot có 2 bậc tự do thay vì 12 và nó bất ổn định hở vòng.

Vì sao viết lại toàn bộ
=======================
Bản trước tự viết lấy gần như mọi thứ: command term riêng, observation riêng, reward riêng,
termination riêng, cộng một cảm biến IMU làm nguồn dữ liệu thứ hai. Hệ quả là các mảnh nói
những điều khác nhau về cùng một robot, và mâu thuẫn nằm rải rác trong chính docstring của
chúng. Những cái đã gỡ:

* ``cover_flat_l2`` tự viết **trùng từng dòng** với ``mdp.flat_orientation_l2`` có sẵn;
* ``BalanceCarVelocityCommand`` (159 dòng) tồn tại chỉ vì lệnh tiến bị đặt vào ``lin_vel_x``
  trong khi xe tiến theo **+Y**. Đặt vào ``lin_vel_y`` là mọi thứ của Isaac Lab đúng ngay:
  reward, metric ``error_vel_xy``, và cả **mũi tên debug** — hàm vẽ mũi tên dùng
  ``atan2(v[1], v[0])`` nên với lệnh ``(0, vy)`` nó chỉ đúng dọc +Y;
* quan sát cũ chỉ đưa vào **một** thành phần vận tốc dài và **một** thành phần vận tốc góc
  (``lin_vel_b[:,1]``, ``ang_vel_b[:,0]``). Policy vì thế **không hề nhìn thấy tốc độ quay
  quanh Z** trong khi vẫn bị chấm điểm theo lệnh ``wz`` — đây là lý do ``error_vel_yaw`` đứng
  ở ~1.2 rad/s trên dải lệnh ±1.0: nó đang được yêu cầu bám một đại lượng nó không quan sát
  được;
* quan sát cũ đưa vào **góc yaw tuyệt đối** — một hướng cố định trong world. Policy học bám
  hướng đó thì không rẽ được. Isaac Lab dùng ``projected_gravity`` thay cho Euler: bắt được cả
  roll lẫn pitch, không có gimbal, và không mang thông tin yaw;
* ``reset_when_fall`` tự viết chỉ đo **roll**, bỏ sót lật ngang; ``mdp.bad_orientation`` đo góc
  giữa trọng lực và trục z nên bắt cả hai trục;
* nhóm ``critic`` giống hệt nhóm ``policy`` từng ký tự — không mua thêm gì;
* IMU gắn ở trục bánh là nguồn dữ liệu thứ hai bên cạnh root state, và hai bên cho số khác
  nhau (trọng tâm cao hơn trục 0.17 m nên khi thân lắc, hai điểm chênh nhau ``l·ω``). Bỏ IMU:
  một nguồn duy nhất là root state, đúng như mọi task locomotion của Isaac Lab.

Kết quả: **không còn một dòng MDP tự viết nào** ở tầng thấp, cả package ``mdp/`` cũ đã xoá.

Quy ước trục — điểm mấu chốt
============================
Hai bánh quay quanh trục world X (xem :mod:`balance_car_cfg`), nên **hướng tiến là +Y của
thân**. Thay vì sửa từng hàm cho hợp trục, ở đây quy ước lệnh được xoay cho khớp:

======================  ==========================================================
``lin_vel_x = (0, 0)``  không có lệnh ngang — xe vi sai không đi ngang được.
                        Thành phần này thành **phạt trượt ngang** miễn phí: bất kỳ
                        vận tốc body-X nào cũng làm ``track_lin_vel`` giảm.
``lin_vel_y``           **lệnh TIẾN/LÙI**. Đây là chỗ khác duy nhất so với mẫu gốc.
``ang_vel_z``           lệnh rẽ, giống mẫu gốc.
======================  ==========================================================

Nhờ vậy ``mdp.track_lin_vel_xy_exp`` (so ``command[:, :2]`` với ``root_lin_vel_b[:, :2]``)
đúng nguyên xi, không cần bọc lại.

Dấu của action cũng suy ra được chứ không phải thử: cả hai khớp cùng trục ``(-1, 0, 0)``, lăn
không trượt cho ``v_tâm = (0, +ωR, 0)``, nên **mô-men dương → tiến (+Y)** và hai ``scale`` cùng
dương. Xem phần suy dẫn trong :mod:`balance_car_cfg`.

Tần số
======
Sim 200 Hz, điều khiển 50 Hz (``decimation = 4``) — giống hệt mẫu locomotion. Bản cũ chạy
60/30 Hz: với hằng số thời gian lật 0.132 s thì 30 Hz chỉ cho **4 bước** mỗi lần biên độ nhân
e, sát mức không cứu kịp. 50 Hz cho 6.6 bước.

Đọc trọng số reward
===================
Isaac Lab nhân reward với ``step_dt`` (``RewardManager.compute``), nên **mọi trọng số ở đây là
tốc độ theo GIÂY, không phải theo bước**. ``track_lin_vel_xy_exp`` trọng số 2.0 nghĩa là tối đa
2.0 điểm mỗi giây, tức 0.04 mỗi bước ở 50 Hz. Đây cũng là lý do
``Episode_Reward/<term>`` trong log đọc được trực tiếp như "điểm trung bình mỗi giây".
"""

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .balance_car_cfg import (
    BALANCE_CAR_CFG,
    BALANCE_CAR_GROUND_FRICTION,
    BALANCE_CAR_TRACTION_TORQUE,
)

##
# Hằng số của bài toán
##

BALANCE_CAR_FALL_ANGLE = math.pi / 180 * 40
"""Ngưỡng ngã [rad] — góc giữa trục z của thân và phương thẳng đứng.

40° khớp với ma sát 1.0: góc lớn nhất còn cứu được thoả ``tan θ = μ``, tức 45°, và ở sát 45°
thì cần vô hạn thời gian. Đặt ngưỡng cao hơn mức khả thi chỉ khiến policy phải học từ những
thế cờ mà vật lý không cho cứu — và không có gì trong log chỉ ra điều đó.
"""

BALANCE_CAR_MAX_SPEED = 1.5
"""Lệnh tiến lớn nhất [m/s].

1.5 m/s = 7.3 rad/s ở bánh 0.205 m, còn xa trần khớp 40 rad/s (8.2 m/s) và xa trần ma sát
(gia tốc tối đa ``μ·g`` = 9.8 m/s²).
"""


##
# Scene
##


@configclass
class BalanceCarSceneCfg(InteractiveSceneCfg):
    """Sàn phẳng + xe. Không có cảm biến nào: root state là nguồn dữ liệu duy nhất."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # "multiply" giống mẫu locomotion: μ hiệu dụng = μ_sàn × μ_bánh, nên phần
                # random ma sát ở EventCfg đi thẳng vào kết quả. Bản cũ dùng "max" — chế độ đó
                # nuốt luôn phần random, mọi env chạy cùng một hệ số.
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=BALANCE_CAR_GROUND_FRICTION,
                dynamic_friction=BALANCE_CAR_GROUND_FRICTION,
                restitution=0.0,
            ),
        ),
    )

    robot: ArticulationCfg = BALANCE_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    """Lệnh vận tốc trong hệ thân.

    Thành phần TIẾN nằm ở ``lin_vel_y`` — xem phần "Quy ước trục" ở đầu module. Nhờ vậy mũi
    tên debug XANH LÁ (lệnh) và XANH DƯƠNG (vận tốc thật) cùng một hệ và chỉ cùng hướng khi
    policy bám tốt; bản cũ vẽ mũi tên lệnh lệch 90° vì lệnh nằm ở ``lin_vel_x``.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        # 20% số env được lệnh đứng yên — giữ bài "cân bằng tại chỗ" trong phân phối
        rel_standing_envs=0.2,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # xe vi sai không đi ngang; để (0,0) biến thành phần này thành phạt trượt ngang
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(-BALANCE_CAR_MAX_SPEED, BALANCE_CAR_MAX_SPEED),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Mô-men hai bánh.

    ``scale`` lấy đúng ngưỡng trượt: quá mức đó bánh chỉ quay trượt chứ không thêm gia tốc,
    nên cho dải rộng hơn chỉ tạo một vùng action mà mọi giá trị cho cùng kết quả và gradient
    bằng 0. Bản cũ để 150 N·m trong khi ngưỡng trượt là 52 — 94% dải action vô dụng.

    Hai khớp cùng ``scale`` dương vì cùng trục world ``(-1, 0, 0)``. Quét lại trục sau mỗi lần
    sinh USD; xem cảnh báo trong :mod:`balance_car_cfg`.
    """

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Revolute_[1-2]"],
        scale=BALANCE_CAR_TRACTION_TORQUE,
    )


@configclass
class ObservationsCfg:
    """Quan sát — đúng bộ của mẫu locomotion, bỏ height scan (sàn phẳng) và bỏ ``joint_pos``.

    Bỏ ``joint_pos`` là có chủ đích: **góc bánh là toạ độ cyclic**, động lực học của xe không
    phụ thuộc vào nó. Nó lại tăng vô hạn khi xe chạy, nên đưa vào mạng là đưa một đại lượng
    không dừng — chuẩn hoá quan sát cũng không cứu được, vì phân phối dịch đi mãi trong lúc
    train. Tốc độ bánh (``joint_vel``) mới là thứ mang thông tin.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Quan sát cho policy. Tên term ``actions`` và ``velocity_commands`` là bắt buộc:
        tầng navigation ghi đè đúng hai term này để tiêm lệnh của nó xuống."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Bật nhiễu quan sát. Bản cũ tắt hẳn: policy khi đó học trên số đo hoàn hảo và
            # gãy ngay khi gặp cảm biến thật, mà cũng dễ bám vào những chi tiết vi mô của
            # solver thay vì vào vật lý.
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Nhiễu miền — phần bản cũ hoàn toàn không có.

    Không có mục nào ở đây thì mọi env là **cùng một con robot trên cùng một mặt sàn**, và
    policy được phép học một nghiệm khớp chính xác với bộ tham số đó. Với xe cân bằng, ba
    tham số dưới đây đúng là ba thứ sai nhiều nhất khi mang sang phần cứng thật.
    """

    # -- startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel", "wheel_01"]),
            # nhân với ma sát sàn (combine "multiply") → μ hiệu dụng 0.7-1.3
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.6, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_frame_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Group_1"),
            # khung 47.1 kg ± 5 → xe phải chịu được tải đặt thêm lên nắp
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    frame_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Group_1"),
            # Lệch trọng tâm là sai số hiệu chỉnh kinh điển của xe cân bằng thật: xe đứng yên
            # ở một góc nghiêng khác 0. ±2 cm trên cánh tay 17 cm là ±6.6° điểm cân bằng.
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # -- reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Thân lật quanh trục X (hai bánh quay quanh X) nên ROLL là bậc tự do cần nhiễu.
            # ±0.15 rad ≈ ±8.6°, còn xa ngưỡng ngã 40° nhưng đủ để policy phải gượng dậy.
            # Không có nhiễu này thì mọi env xuất phát ở đúng một điểm cân bằng hoàn hảo.
            "pose_range": {"yaw": (-math.pi, math.pi), "roll": (-0.15, 0.15), "pitch": (-0.05, 0.05)},
            "velocity_range": {"y": (-0.3, 0.3), "roll": (-0.4, 0.4)},
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_[1-2]"]),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # -- interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Toàn bộ là term có sẵn của Isaac Lab. Trọng số là **điểm mỗi giây** (xem đầu module)."""

    # -- nhiệm vụ
    # Bám lệnh phải là tín hiệu lớn nhất, nếu không nghiệm rẻ nhất luôn là đứng im.
    # std 0.5 chứ không phải 0.25: với dải lệnh ±1.5 mà để std 0.25 thì sai số 1 m/s cho
    # exp(-16) ≈ 0 — lúc mới train phần lớn env nằm ở vùng reward phẳng lì, không có gradient
    # nào chỉ đường.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- giữ nắp xe phẳng
    # Bằng đúng ``cover_flat_l2`` tự viết của bản cũ: sin² của độ nghiêng, đo bằng
    # projected_gravity nên bắt cả roll lẫn pitch, còn dốc ở mọi góc.
    #
    # KHÔNG đặt nặng hơn: xe cân bằng **bắt buộc phải nghiêng để tăng tốc**, độ nghiêng chính
    # là tín hiệu điều khiển (``θ = atan(a/g)``). Bản cũ chồng thêm một term thưởng nhọn ±3°
    # trọng số 3.5 lên trên; khi đó chỉ cần gia tốc 0.5 m/s² là mất nhiều điểm hơn cả phần
    # thưởng bám lệnh tối đa, và "đứng im, phẳng lì" trở thành nghiệm TỐI ƯU. Đúng thứ policy
    # đã học. Ở -2.0 thì nghiêng 5.8° (gia tốc 1 m/s²) chỉ tốn 0.02 điểm/giây.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # Damp nhịp lắc quanh trục ngang — đây mới là term làm nắp xe hết "rung lag" trong video,
    # chứ không phải phạt góc mạnh hơn.
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # -- lực bánh phải mượt
    # Không có ba term này thì nghiệm rẻ nhất của policy là bang-bang: đảo mô-men hết biên mỗi
    # bước. Vừa xấu khi quay video vừa không chuyển được sang phần cứng thật.
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- ngã
    # Chỉ nổ khi ngã thật: ``time_out`` được đánh dấu ``time_out=True`` nên là TRUNCATION và
    # ``is_terminated`` không tính nó. Trọng số nhân với dt → -100 × 0.02 = -2.0 một lần, xấp
    # xỉ 1 giây bám lệnh hoàn hảo. Phần phạt chính vẫn là mất toàn bộ reward tương lai.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Góc giữa trục z của thân và phương thẳng đứng — bắt cả lật trước/sau lẫn lật ngang.
    # Bản cũ tự viết bằng Euler roll nên bỏ sót hoàn toàn lật ngang.
    base_fell = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": BALANCE_CAR_FALL_ANGLE})


##
# Env
##


@configclass
class BalanceCarEnvCfg(ManagerBasedRLEnvCfg):
    """Xe hai bánh tự cân bằng bám lệnh vận tốc."""

    scene: BalanceCarSceneCfg = BalanceCarSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # 200 Hz sim, 50 Hz điều khiển — giống mẫu locomotion. Xem phần "Tần số" ở đầu module.
        self.decimation = 4
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        # 4 lệnh mỗi episode ở resampling 5 s
        self.episode_length_s = 20.0
        # vật liệu sàn cũng là vật liệu mặc định của scene, để mọi va chạm khác cùng hệ số
        self.sim.physics_material = self.scene.ground.spawn.physics_material

        # xe tiến dọc +Y nên camera phải lệch sang X mới thấy được chuyển động tiến/lùi
        self.viewer.eye = (8.0, -4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)


@configclass
class BalanceCarEnvCfg_PLAY(BalanceCarEnvCfg):
    """Bản để xem/quay video: ít env, không nhiễu, không xô đẩy."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # giữ nguyên phần random ma sát/khối lượng: chính nó cho thấy policy có thật sự bền
        # hay chỉ khớp với một bộ tham số
