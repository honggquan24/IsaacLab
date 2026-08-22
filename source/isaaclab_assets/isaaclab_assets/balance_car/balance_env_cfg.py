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
from .balance_car_cfg import (
    BALANCE_CAR_AXLE_OFFSET,
    BALANCE_CAR_CFG,
    BALANCE_CAR_GROUND_FRICTION,
    BALANCE_CAR_TRACTION_TORQUE,
)
from .mdp.commands import BalanceCarVelocityCommandCfg
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
        spawn=sim_utils.GroundPlaneCfg(
            # Ma sát PHẢI đặt tay: mặc định của Isaac Lab là 0.5, USD Onshape không mang vật
            # liệu vật lý nào, và ở 0.5 thì xe không cứu nổi độ nghiêng quá 26.6° — triệu chứng
            # nhìn ra là "bánh yếu" trong khi thật ra là bánh TRƯỢT. Xem BALANCE_CAR_GROUND_FRICTION.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=BALANCE_CAR_GROUND_FRICTION,
                dynamic_friction=BALANCE_CAR_GROUND_FRICTION * 0.9,
                restitution=0.0,
                # "max" có ưu tiên cao nhất trong PhysX nên giá trị này thắng, không bị lấy
                # trung bình với 0.5 mặc định của bánh
                friction_combine_mode="max",
            ),
        ),
    )

    # Add robot
    robot: Articulation = BALANCE_CAR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    # IMU gắn trên khung. Đường dẫn đổi theo bản CAD mới: thân tên `Group_1`, nằm dưới
    # `robot/robot` chứ không phải `balance_robot/balance_robot`.
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/robot/robot/Group_1",
        offset=ImuCfg.OffsetCfg(
            # gốc thân ở MẶT TRÊN khung; dời IMU xuống đúng trục bánh để `lin_vel_b` là vận
            # tốc tiến thật của xe, không lẫn phần vung của đỉnh khung khi thân nghiêng
            pos=(0.0, 0.0, BALANCE_CAR_AXLE_OFFSET),
            # bản CAD mới đã vẽ thẳng nên IMU không phải xoay bù. Bản cũ để (0,0,0,1) tức
            # xoay 180° quanh Z — chính phép xoay đó ĐỔI DẤU lin_vel_b[:,1] và ang_vel_b[:,0].
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        update_period=0.0,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Mô-men hai bánh.

    ``scale`` lấy ``BALANCE_CAR_TRACTION_TORQUE`` (~35 N·m) làm trần: quá mức đó bánh chỉ
    quay trượt tại chỗ chứ không thêm gia tốc, nên cho policy dải rộng hơn là cho nó một
    vùng action vô dụng để mò.

    .. important::
        Hai ``scale`` CÙNG DƯƠNG, và điều đó phụ thuộc vào bản export. Bản USD hiện tại cho
        cả hai khớp trục world ``(-1, 0, 0)`` nên cùng dấu là cùng chiều quay, và
        "hai output cùng dương = xe tiến theo +Y". Bản export trước của cùng file CAD lại cho
        hai trục ngược nhau, lúc đó ``Revolute_2`` phải mang dấu âm. Quét lại trục khớp sau
        mỗi lần sinh USD — xem ``BALANCE_CAR_CFG``.
    """

    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Revolute_1", "Revolute_2"],
        scale={
            "Revolute_1": BALANCE_CAR_TRACTION_TORQUE,
            "Revolute_2": BALANCE_CAR_TRACTION_TORQUE,
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

    # Dùng bản kế thừa để metric đo ĐÚNG TRỤC — bản gốc so lệnh tiến với vận tốc ngang vì xe
    # này tiến theo +Y của body chứ không phải +X. Xem mdp/commands.py.
    base_velocity = BalanceCarVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        # 20% số env được lệnh đứng yên — giữ nguyên bài "cân bằng tại chỗ" trong phân phối
        rel_standing_envs=0.2,
        heading_command=False,
        debug_vis=True,
        ranges=BalanceCarVelocityCommandCfg.Ranges(
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
    """Nhiễu lúc reset.

    Bánh xe quay tự do nên xê dịch góc bánh gần như không đổi gì bài toán — thứ thật sự tạo
    ra sự đa dạng là **độ nghiêng ban đầu của thân**. Không có nó thì mọi env xuất phát ở
    đúng một trạng thái cân bằng hoàn hảo và policy không bao giờ gặp tình huống phải gượng.
    """

    reset_base = EventTerm(
        func=mdp.events.reset_root_state_uniform,
        mode="reset",
        params={
            # xe lật quanh trục X (hai bánh quay quanh X) nên roll là bậc tự do cần nhiễu.
            # ±0.1 rad ≈ ±5.7°, còn xa ngưỡng ngã 50° của `reset_when_fall`.
            "pose_range": {"yaw": (-math.pi, math.pi), "roll": (-0.1, 0.1)},
            "velocity_range": {"roll": (-0.2, 0.2)},
        },
    )
    reset_wheels = EventTerm(
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

    # NẮP XE PHẢI NẰM NGANG TUYỆT ĐỐI — hai term bổ cho nhau, xem chú thích trong mdp/rewards.py.
    # Thay cho reward_angle_r + bonus_reward cũ: hai cái đó chỉ đo ROLL (bỏ sót pitch), dùng
    # cos nên phẳng lì quanh 0, và cái bonus là bậc thang cứng ±2° không có đạo hàm để bám.
    #
    # cover_flat  = sin²(độ nghiêng), phạt rộng — còn dốc ở mọi góc nên luôn kéo về phẳng.
    #               Ở 10° đóng -0.15/bước, ở 40° (ngưỡng ngã) là -2.07/bước.
    cover_flat = RewardTermCfg(func=cover_flat_l2, weight=-5.0)
    # cover_flat_bonus = đỉnh nhọn rộng ~±3°, lo vài độ cuối mà term L2 đã hết lực kéo.
    # std 0.12 (~7°) chứ KHÔNG phải 0.05 (~3°), và trọng số 1.0 chứ không phải 3.5.
    #
    # Xe cân bằng BẮT BUỘC phải nghiêng để tăng tốc — độ nghiêng chính là tín hiệu điều khiển,
    # ``θ = atan(a/g)``. Với đỉnh nhọn ±3° và trọng số 3.5, chỉ cần gia tốc 0.5 m/s² (nghiêng
    # 2.9°) là mất 2.24 điểm/giây, trong khi bám lệnh giỏi nhất cũng chỉ được 2.0. Đứng im và
    # phẳng lì trở thành nghiệm TỐI ƯU — đúng cái policy đã học ở vòng 256.
    #
    # Ở std 0.12 và trọng số 1.0 thì nghiêng 2.9° chỉ tốn 0.17/giây, so với 4.0 điểm bám lệnh.
    cover_flat_bonus = RewardTermCfg(func=cover_flat_exp, weight=1.0, params={"std": 0.12})
    reward_r_rate = RewardTermCfg(func=reward_roll_rate, weight=0.4)

    # LỰC BÁNH PHẢI MƯỢT. Trước đây không có term nào phạt độ giật, nên nghiệm rẻ nhất của
    # policy là bang-bang: đảo mô-men hết biên mỗi bước. Vừa xấu khi quay video vừa không
    # chuyển được sang phần cứng thật.
    #   action_rate: đảo action từ -1 sang +1 trên cả hai bánh cho ra -0.02·8 = -0.16/bước.
    #                Đừng nâng quá tay — xe có hằng số thời gian lật 0.132 s, chặn phản ứng
    #                nhanh là chặn luôn khả năng cân bằng.
    action_rate = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.02)
    #   joint_torque: phạt độ lớn mô-men, đẩy policy về nghiệm ít tốn lực. Ở mức bão hoà
    #                 2 × 33² = 2178 thì đóng -0.044/bước.
    joint_torque = RewardTermCfg(func=rewards.joint_torques_l2, weight=-2.0e-5)

    # bám lệnh
    # Nâng từ 2.0/1.0: bám lệnh phải là term LỚN NHẤT, nếu không policy sẽ chọn đứng im cho
    # rẻ. Ở vòng 256 cover_flat_bonus (3.42) một mình đã lớn hơn tổng hai term này (2.08).
    track_lin_vel = RewardTermCfg(func=track_lin_vel_exp, weight=4.0, params={"std": 0.25})
    track_ang_vel = RewardTermCfg(func=track_ang_vel_exp, weight=2.0, params={"std": 0.5})


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
        env_spacing=5.0,
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
        # xe tiến dọc Y nên đặt camera lệch sang X mới thấy được chuyển động tiến/lùi
        self.viewer.eye = (15.5, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)

        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps
