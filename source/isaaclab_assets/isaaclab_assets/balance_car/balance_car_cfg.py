# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cấu hình robot xe hai bánh tự cân bằng.

File này chỉ mô tả **con robot**, không mô tả bài toán — đúng vai trò của
``isaaclab_assets/robots/anymal.py`` trong Isaac Lab. Phần môi trường nằm ở
:mod:`balance_env_cfg`, dựng theo mẫu ``isaaclab_tasks/manager_based/locomotion/velocity``.

Mọi con số dưới đây **đọc thẳng từ USD**, không ước lượng
--------------------------------------------------------
Bản trước của file này ghi ``mass = 27.45 kg`` trong khi USD khai báo **52.13 kg**, và ghi
"đọc thẳng từ USD" trong khi USD lúc đó chưa hề được đọc. Sai gần gấp đôi khối lượng làm sai
theo mọi thứ dẫn xuất: mô-men trượt, mô-men gượng dậy, cỡ của action. Bảng này lấy từ
``UsdPhysics.MassAPI`` của từng mesh trong ``usd/balance_car_cfg.usd``:

======================  ==========  =================================================
thân                    khối lượng  trọng tâm (world, gốc thân = mặt trên khung)
======================  ==========  =================================================
``base``                36.537 kg   z = −0.3177
``cover``               10.603 kg   z = −0.0100
``wheel`` (mỗi bánh)     2.495 kg   z = −0.4200, x = ±0.196
**tổng**                **52.130**  z = −0.2649
======================  ==========  =================================================

Hình học
--------
* khung ``Group_1``: ``base`` là hộp 0.30 × 0.30 × 0.40 m (z ∈ [−0.42, −0.02]), ``cover`` là
  tấm 0.50 × 0.50 × 0.02 m trên đỉnh. **Gốc thân nằm ở mặt trên nắp**, nên mọi cao độ trong
  hệ thân đều âm — đó là lý do ``init_state.pos`` phải bù lên;
* trục bánh ở z = −0.42, tâm bánh x = ±0.196 (vệt bánh 0.392 m); bánh bán kính 0.2050 m,
  dày 0.09 m → đáy bánh z = −0.625, thả ở z = 0.63 là vừa chạm sàn với 5 mm hở;
* va chạm dùng ``convexHull`` (bao lồi của lưới), kể cả bánh xe. Bánh vì thế là đa giác
  88 đỉnh chứ không phải hình tròn trơn — lăn có gợn nhẹ ở tốc độ thấp. Muốn tròn thật thì
  phải đổi approximation trong CAD hoặc thay bằng primitive cylinder.

Con lắc ngược của bài toán
--------------------------
Cái lật là **khung**, quay quanh trục bánh. Trọng tâm khung nằm cao hơn trục 0.1715 m
(``BALANCE_CAR_COM_HEIGHT``), nên hằng số thời gian lật là ``√(l/g) = 0.132 s``. Ở tần số
điều khiển 50 Hz đó là **6.6 bước** cho mỗi lần biên độ nhân e — đủ để một mạng phản ứng kịp.
Ở 30 Hz chỉ còn 4 bước, sát mức không cứu nổi; đó là lý do :mod:`balance_env_cfg` chạy sim
200 Hz / điều khiển 50 Hz thay vì 60/30 như bản cũ.

Chiều quay và hướng tiến — SUY RA ĐƯỢC, KHÔNG PHẢI ĐOÁN
-------------------------------------------------------
Quét USD hiện tại cho **cả hai khớp cùng trục world (−1, 0, 0)**. Từ đó suy ra chiều tiến
bằng động học lăn không trượt, không cần nhìn mô phỏng:

    ω = (−ω, 0, 0),  r (tâm → điểm tiếp xúc) = (0, 0, −R)
    v_tiếp_xúc = v_tâm + ω × r = 0   ⟹   v_tâm = −(ω × r) = (0, +ωR, 0)

Vậy **mô-men dương → xe chạy theo +Y của thân**, và vì hai khớp cùng trục nên **cả hai
``scale`` cùng dấu dương**. Đây là lý do quy ước lệnh trong :mod:`balance_env_cfg` đặt thành
phần tiến vào ``lin_vel_y`` chứ không phải ``lin_vel_x``.

.. warning::
    Onshape ghi khung khớp theo thứ tự người vẽ chọn mate, nên **chiều trục đổi giữa các lần
    export kể cả khi hình học không đổi**. Một bản export trước của chính file CAD này cho
    ``Revolute_1 = (−1,0,0)`` nhưng ``Revolute_2 = (+1,0,0)``; để nguyên hai scale dương lúc
    đó thì hai bánh quay ngược nhau, xe xoay tại chỗ mà log vẫn đẹp. Sau **mỗi** lần sinh USD
    phải quét lại trục world của hai khớp: cùng dấu → hai scale cùng dương, khác dấu → một
    trong hai phải âm.

Chuẩn bị USD
------------
``usd/balance_car_base.usd`` là bản Onshape thô, ``usd/balance_car_cfg.usd`` là bản đã vá và
là bản env dùng. Bản thô thiếu ``DriveAPI`` (actuator không sinh được mô-men), có
``body0``/``body1`` ngược, và không có trần tốc độ khớp. Sinh lại bằng::

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

##
# Hình học và khối lượng — đọc từ USD, xem bảng trong docstring module.
##

BALANCE_CAR_WHEEL_RADIUS = 0.2050
"""Bán kính bánh [m]."""

BALANCE_CAR_TRACK_WIDTH = 0.392
"""Khoảng cách giữa tâm hai bánh [m]. Quyết định mô-men rẽ: quay ở ``wz`` cần chênh lệch
lực hai bánh ``ΔF = I_z·ẇz / track``."""

BALANCE_CAR_AXLE_OFFSET = -0.42
"""Cao độ trục bánh so với gốc thân ``Group_1`` [m]. Gốc thân ở mặt trên nắp nên số này âm."""

BALANCE_CAR_SPAWN_HEIGHT = 0.63
"""Cao độ thả xe [m] = |AXLE_OFFSET| + bán kính bánh + 5 mm hở cho solver."""

BALANCE_CAR_MASS = 52.130
"""Tổng khối lượng [kg] — khung 47.140 (base 36.537 + cover 10.603) + hai bánh 4.990."""

BALANCE_CAR_COM_HEIGHT = 0.1715
"""Trọng tâm KHUNG so với trục bánh [m] — chiều dài con lắc ngược thật của bài toán.

Dùng trọng tâm khung chứ không phải trọng tâm cả xe (0.1551 m): bánh xe quay quanh trục nên
không góp phần vào mô-men lật, chỉ khung mới lật.
"""

##
# Ma sát và mô-men — hai số này PHẢI đi cùng nhau.
##

BALANCE_CAR_GROUND_FRICTION = 1.0
"""Hệ số ma sát danh nghĩa bánh–sàn.

**Phải đặt tay.** Mặc định của :class:`RigidBodyMaterialCfg` là 0.5 và USD từ Onshape không
mang vật liệu vật lý nào, nên để mặc định là xe chạy trên sàn trơn — triệu chứng nhìn ra là
"bánh xe quá yếu" trong khi thật ra bánh đang **trượt**.

Giới hạn nằm ở ma sát chứ không ở mô-men. Gia tốc lớn nhất là ``μ·g`` và góc nghiêng lớn nhất
còn cứu được thoả ``tan θ = μ``:

======  ==================  ===============  ==========================
μ       mô-men trượt/bánh   gia tốc tối đa   góc cứu được tối đa
======  ==================  ===============  ==========================
0.5     26.2 N·m            4.9 m/s²         26.6°  ← mặc định, quá trơn
1.0     52.4 N·m            9.8 m/s²         **45.0°**  ← đang dùng
1.2     62.9 N·m            11.8 m/s²        50.2°
======  ==================  ===============  ==========================

Chọn 1.0 để **khớp với ngưỡng ngã 40°** của termination: mọi tư thế mà env còn cho tồn tại
đều là tư thế vật lý cứu được. Đặt ma sát thấp hơn ngưỡng ngã (bản cũ để 0.3 → chỉ cứu được
16.7°) là bắt policy học từ những thế cờ vật lý không cho phép cứu, và không log nào chỉ ra
điều đó.

Giá trị này đi vào ground plane với ``friction_combine_mode="multiply"``, còn ma sát của bánh
được random quanh 1.0 trong :class:`EventCfg` — tích của hai bên cho μ hiệu dụng 0.7–1.3.
"""

BALANCE_CAR_TRACTION_TORQUE = BALANCE_CAR_GROUND_FRICTION * (BALANCE_CAR_MASS * 9.81 / 2) * BALANCE_CAR_WHEEL_RADIUS
"""Mô-men [N·m] mà một bánh bắt đầu trượt. Hiện ≈ **52.4**.

Tính từ ``μ · (m·g/2) · R`` chứ không gõ số cứng: chỉnh ma sát thì trần mô-men tự đi theo.
Bản cũ ghi cứng 150.0 bên cạnh một docstring nói rằng nó được tính ra ≈ 33 — tức action scale
lớn gấp 18 lần ngưỡng trượt thật, và 94% dải action là vùng mà mọi giá trị cho cùng một kết
quả vật lý. Gradient trong vùng đó bằng 0; policy vẫn phải mò trong đó.

Đối chiếu độ lớn để thấy 52.4 là đủ:

* giữ TĨNH ở 10°: ``m·g·l·sin10° = 15.2 N·m`` cho cả hai bánh → 7.6 mỗi bánh;
* **gượng dậy từ 30°** (con số định cỡ): cần gia tốc ``g·tan30° = 5.66 m/s²``, tức
  ``m·a·R/2 = 30.3 N·m`` mỗi bánh — còn dư 42% trước khi trượt.
"""

BALANCE_CAR_MAX_WHEEL_SPEED = 40.0
"""Trần tốc độ khớp [rad/s], khớp với ``physxJoint:maxJointVelocity`` ghi trong USD.

40 rad/s × 0.205 m = 8.2 m/s. Để actuator thấp hơn giá trị trong USD thì actuator tự cắt
trước — lại thêm một kiểu "bánh yếu" nữa mà log không nói.
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
        # Hai bánh cùng trục world (-1,0,0) nên nhóm chung một actuator: cùng mô-men, cùng
        # trần, cùng dấu. Tách "trái"/"phải" thành hai ImplicitActuatorCfg như bản cũ không
        # mua thêm gì mà lại tạo chỗ để hai bên lệch tham số theo thời gian.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_[1-2]"],
            # trần cứng gấp 2 lần ngưỡng trượt: policy Gaussian xuất ra ngoài [-1,1] là bình
            # thường, trần này chỉ để lệnh biên không phá solver
            effort_limit_sim=2.0 * BALANCE_CAR_TRACTION_TORQUE,
            velocity_limit_sim=BALANCE_CAR_MAX_WHEEL_SPEED,
            # điều khiển thuần mô-men: không có vòng vị trí/vận tốc nào của PhysX xen vào
            stiffness=0.0,
            damping=0.0,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        # thân nổi: Isaac Lab đặt tư thế thân gốc theo đúng giá trị này và KHÔNG đọc phép tịnh
        # tiến trong USD — chiều cao phải bù ở đây, nâng trong file USD không có tác dụng
        pos=(0.0, 0.0, BALANCE_CAR_SPAWN_HEIGHT),
        # quaternion đơn vị = tư thế đứng; bản CAD hiện tại đã vẽ thẳng nên không phải xoay bù
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"Revolute_[1-2]": 0.0},
        joint_vel={"Revolute_[1-2]": 0.0},
    ),
)
"""Xe hai bánh tự cân bằng, thân nổi, hai bánh điều khiển bằng mô-men.

Hướng tiến là **+Y của thân** — xem phần suy dẫn trong docstring module.
"""
