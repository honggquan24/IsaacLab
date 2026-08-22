# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cấu hình xe hai bánh tự cân bằng (CAD Onshape, bản vẽ lại).

Số liệu dưới đây đọc thẳng từ ``usd/balance_car_base.usd`` để khỏi phải mở stage mới biết.
Bản vẽ mới KHÁC HẲN bản cũ, không chỉ to hơn:

============================  =========================  =========================
đại lượng                     bản cũ (balance_car_v1)    bản mới (balance_car)
============================  =========================  =========================
prim articulation             ``balance_robot``          ``robot``
thân khung                    ``balance_body``           ``Group_1``
tư thế đứng                   ``|roll| = 90°``           ``roll = 0°``
``Revolute_1``                bánh x = **-0.115** (trái) bánh x = **+0.15** (phải)
``Revolute_2``                bánh x = **+0.115** (phải) bánh x = **-0.15** (trái)
bán kính bánh                 0.05 m                     0.2050 m
khối lượng                    (không ghi)                27.45 kg
============================  =========================  =========================

Hình học bản mới
----------------
* khung ``Group_1`` là hộp 0.5 × 0.5 × 0.42 m, **gốc thân nằm ở MẶT TRÊN** của khung: mesh
  trải z ∈ [-0.42, 0] so với gốc đó. Đây là lý do ``init_state.pos`` phải bù ngược lên;
* trục bánh ở z = -0.42 (so với gốc thân), x = ±0.15; bánh bán kính 0.2050 m, dày 0.09 m nên
  đáy bánh ở z = -0.6250 → **thả xe ở z = 0.63 là bánh vừa chạm sàn**;
* khối lượng: khung 22.46 kg (base 11.86 + cover 10.60), mỗi bánh 2.50 kg. Tổng 27.45 kg.
  Trọng tâm nằm ở z = -0.248, tức **cao hơn trục bánh 0.172 m** — đây là chiều dài con lắc
  ngược thật sự của bài toán, và nó quyết định bài dễ hay khó: hằng số thời gian lật
  ``√(l/g) = 0.132 s``, ở 30 Hz là **4 bước điều khiển** cho mỗi lần biên độ nhân e. Nếu train
  mãi không cân được thì nút gỡ đầu tiên là ``decimation = 1`` (60 Hz);
* trục quay của **cả hai bánh là trục X của world**, nên xe tiến/lùi dọc **Y** và rẽ quanh
  **Z**. Tư thế đứng ứng với quaternion đơn vị, tức ``roll = pitch = yaw = 0``.

Chiều trục khớp — PHẢI KIỂM LẠI SAU MỖI LẦN EXPORT
--------------------------------------------------
Bản USD hiện tại cho **cả hai khớp cùng trục world (-1, 0, 0)**, nên hai bánh quay cùng chiều
khi nhận cùng dấu, và ``ActionsCfg`` để ``scale`` DƯƠNG cho cả hai.

Đừng coi đó là hằng số của robot. Một bản export TRƯỚC của **cùng file CAD này** cho
``Revolute_1 = (-1, 0, 0)`` nhưng ``Revolute_2 = (+1, 0, 0)`` — ngược nhau, và lúc đó
``Revolute_2`` bắt buộc phải mang ``scale`` âm. Onshape ghi khung khớp theo thứ tự người vẽ
chọn mate, nên chiều trục **đổi giữa các lần import** kể cả khi hình học không đổi. Để sai thì
hai bánh quay ngược nhau, xe xoay tại chỗ mà log vẫn đẹp — đúng lỗi đã tốn nhiều thời gian ở
robot V5.

Kiểm sau mỗi lần sinh USD: quét trục world của hai khớp, hai giá trị phải **giống dấu nhau**.
Cùng dấu → hai ``scale`` cùng dương. Khác dấu → một trong hai phải âm.

Một bẫy nữa: ``Revolute_1``/``Revolute_2`` đã **đổi bên** so với bản CAD cũ. Mọi hằng số
trái/phải chép từ code cũ đều sai.

Chuẩn bị USD
------------
``usd/balance_car_base.usd`` là bản Onshape thô, ``usd/balance_car_cfg.usd`` là bản đã vá và
là bản env dùng. Bản thô còn ba lỗi export quen thuộc: ``body0``/``body1`` ngược (bánh làm cha
của khung, khiến khung có hai cha — cây articulation không hợp lệ), thiếu ``DriveAPI`` nên
actuator không sinh được mô-men, và không có trần tốc độ khớp. Sinh lại bằng::

    ./isaaclab.sh -p scripts/ute/prepare_usd.py --package balance_car \\
        --floating-base --base-body Group_1 --max-angular-velocity 40 --verify
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BALANCE_CAR_USD_PATH = os.path.join(CURRENT_DIR, "usd", "balance_car_cfg.usd")
"""USD đã vá. Bản Onshape thô nằm cạnh nó ở ``balance_car_base.usd``."""

BALANCE_CAR_WHEEL_RADIUS = 0.2050
"""Bán kính bánh [m], đo từ mesh trong USD."""

BALANCE_CAR_AXLE_OFFSET = -0.42
"""Cao độ trục bánh so với gốc thân ``Group_1`` [m]. Gốc thân nằm ở mặt trên khung."""

BALANCE_CAR_SPAWN_HEIGHT = 0.63
"""Cao độ thả xe [m] = |AXLE_OFFSET| + bán kính bánh + ~5 mm hở cho solver."""

BALANCE_CAR_MASS = 27.45
"""Tổng khối lượng [kg]: khung 22.46 (base 11.86 + cover 10.60) + hai bánh 4.99."""

BALANCE_CAR_COM_HEIGHT = 0.172
"""Chiều cao trọng tâm so với trục bánh [m] — chiều dài con lắc ngược của bài toán."""

BALANCE_CAR_GROUND_FRICTION = 0.3
"""Hệ số ma sát tĩnh bánh–sàn, đặt ở vật liệu của ground plane.

**Phải đặt tay.** Mặc định của Isaac Lab (:class:`RigidBodyMaterialCfg`) là **0.5** cho cả hai
mặt, mà USD từ Onshape thì không mang vật liệu vật lý nào — nên nếu để mặc định thì xe chạy
trên sàn trơn như băng, và triệu chứng là "bánh xe quá yếu": bơm bao nhiêu mô-men cũng chỉ
quay trượt tại chỗ.

Giới hạn thật nằm ở ma sát chứ không ở mô-men. Gia tốc lớn nhất mà xe đạt được là ``μ·g``, và
góc nghiêng lớn nhất còn cứu được thoả ``tan θ = μ``:

======  ==================  =================  ===========================
μ       mô-men trượt/bánh   gia tốc tối đa     góc cứu được tối đa
======  ==================  =================  ===========================
0.5     13.8 N·m            4.9 m/s²           **26.6°** ← mặc định, quá trơn
1.0     27.6 N·m            9.8 m/s²           45.0°
1.2     33.1 N·m            11.8 m/s²          **50.2°** ← đang dùng
======  ==================  =================  ===========================

Ở 0.5 thì mọi độ nghiêng quá 26.6° là vô phương cứu — không phải policy dở mà là vật lý không
cho. 1.2 vẫn nằm trong khoảng thật của cao su bám trên bê tông nhám (lốp đua lên tới ~1.5), và
nó đẩy trần góc cứu được lên trên ngưỡng ngã 40°, tức policy còn chỗ xoay xở ở mọi trạng thái
mà env cho phép tồn tại.
``friction_combine_mode="max"`` để giá trị này thắng, chứ chế độ ``average`` mặc định sẽ trung
bình với 0.5 của bánh thành 0.85.
"""

BALANCE_CAR_TRACTION_TORQUE = 150.0
"""Mô-men [N·m] mà một bánh bắt đầu trượt, ở ``BALANCE_CAR_GROUND_FRICTION``. Hiện ≈ 33.1.

Tính thẳng từ ``μ · (m·g/2) · R`` chứ không gõ số cứng: chỉnh ma sát thì trần mô-men tự đi
theo, khỏi phải nhớ sửa hai chỗ.

Trên mức này bánh **trượt**. Đẩy ``scale`` cao hơn không mua thêm được gia tốc, chỉ để lại
một vùng action mà mọi giá trị đều cho cùng một kết quả — policy vẫn phải mò trong đó, và
gradient ở đó bằng 0. Nếu thấy "bánh yếu" thì thủ phạm gần như luôn là ma sát, không phải
mô-men: xem bảng trong ``BALANCE_CAR_GROUND_FRICTION``.

Có một ngoại lệ đáng biết: kể cả lúc trượt, mô-men bánh vẫn sinh **phản mô-men** lên khung
(bánh xe làm reaction wheel), nên về lý thuyết mô-men vượt ngưỡng trượt không hoàn toàn vô
dụng cho việc chỉnh tư thế. Nhưng đó là cách cân bằng suy biến — xe quay bánh tại chỗ để giữ
thân — không phải hành vi mình muốn dạy, nên vẫn chặn ở ngưỡng trượt.

Đối chiếu độ lớn: giữ thân TĨNH ở 10° chỉ cần ``m·g·l·sin10° = 8.0 N·m`` cho cả hai bánh. Nhưng
giữ tĩnh không phải bài toán — **gượng dậy từ 30°** đòi xe tăng tốc ``g·tan30° = 5.7 m/s²``,
tức ``m·a·R/2 = 15.9 N·m`` mỗi bánh. Đó mới là con số định cỡ.
"""

BALANCE_CAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BALANCE_CAR_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # bánh lồng sát khung, bật self-collision là rung
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    actuators={
        # Revolute_1 = bánh x = +0.15. Với hướng tiến +Y và hướng lên +Z thì +X là bên PHẢI.
        "wheel_right": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],
            # trần cứng, gấp 2 lần ngưỡng trượt — chỉ để lệnh lỗi không phá solver
            effort_limit_sim=2.0 * BALANCE_CAR_TRACTION_TORQUE,
            # khớp với physxJoint:maxJointVelocity ghi trong USD (40 rad/s = 8.2 m/s); để thấp
            # hơn là actuator tự cắt trước, đúng một kiểu "bánh yếu" nữa
            velocity_limit_sim=40.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # Revolute_2 = bánh x = -0.15, bên TRÁI. Đổi bên so với bản cũ.
        "wheel_left": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_2"],
            effort_limit_sim=2.0 * BALANCE_CAR_TRACTION_TORQUE,
            velocity_limit_sim=40.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        # thân nổi: Isaac Lab đặt tư thế thân gốc theo đúng giá trị này, KHÔNG đọc phép tịnh
        # tiến trong USD — nên chiều cao phải bù ở đây chứ không nâng trong file USD
        pos=(0.0, 0.0, BALANCE_CAR_SPAWN_HEIGHT),
        # quaternion đơn vị = tư thế đứng; bản CAD mới đã vẽ thẳng nên không phải xoay bù
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"Revolute_1": 0.0, "Revolute_2": 0.0},
        joint_vel={"Revolute_1": 0.0, "Revolute_2": 0.0},
    ),
)
"""Xe hai bánh tự cân bằng, thân nổi, chỉ hai bánh được điều khiển."""
