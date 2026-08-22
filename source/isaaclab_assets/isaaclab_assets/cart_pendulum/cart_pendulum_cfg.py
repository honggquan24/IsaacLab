# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cấu hình robot con lắc đơn trên xe đẩy (CAD Onshape).

Hình học lấy từ ``usd/cart_pendulum_cfg.usd``, sinh ra bởi
``scripts/ute/cart_pendulum/prepare_usd.py``. Vài số đo đọc thẳng từ USD, ghi lại ở đây để
khỏi phải mở stage mới biết:

* ray dài 1.11 m nằm dọc trục **Y** của world, giới hạn khớp trượt ±0.555 m;
* xe là khối 50 mm, tâm ở giữa ray khi ``Slider_1`` bằng 0;
* con lắc dài 0.22 m và **thõng xuống** khi ``Revolute_1`` bằng 0 — tư thế đứng là π, nên
  ``init_state.joint_pos`` đặt ``Revolute_1`` = π để mặc định của robot chính là tư thế cần giữ;
* cả cụm đã được nâng trong USD cho đáy con lắc cách sàn 4 cm, vì vậy ``init_state.pos``
  để (0, 0, 0) chứ không phải bù bằng tay như trước.

``rack`` được neo vào world bằng ``FixedJoint`` ngay trong USD nên đây là articulation nền
cố định; không cần ``fix_root_link``, Isaac Lab tự nhận ra khớp đó. Cũng không đặt
``articulation_root_prim_path``: bỏ trống thì Isaac Lab tự dò prim mang ``ArticulationRootAPI``
dưới prim spawn, nhờ vậy đổi tên document bên Onshape cũng không phải sửa gì ở đây.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CART_PENDULUM_USD_PATH = os.path.join(CURRENT_DIR, "usd", "cart_pendulum_cfg.usd")
"""USD đã vá. Bản Onshape thô nằm cạnh nó ở ``cart_pendulum_base.usd``."""

CART_PENDULUM_RAIL_LIMIT = 0.555
"""Nửa chiều dài ray [m], đúng bằng giới hạn khớp ``Slider_1`` trong USD."""

CART_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CART_PENDULUM_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # chặn thêm ở mức thân, cao hơn trần khớp một chút để trần khớp mới là cái ràng buộc
            max_linear_velocity=20.0,
            max_angular_velocity=30.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # xe trượt lồng trong ray nên hai thân luôn chạm nhau; bật self-collision là rung
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    actuators={
        "cart": ImplicitActuatorCfg(
            joint_names_expr=["Slider_1"],
            # xe + con lắc nặng cỡ 0.15 kg (PhysX tự tính từ convex hull, khối lượng riêng
            # mặc định 1000 kg/m³). Lực đẩy do action quyết định (scale bên ActionsCfg), giới
            # hạn 100 N ở đây chỉ là trần cứng để lệnh lỗi không văng xe đi.
            effort_limit_sim=100.0,
            # trần khớp cũng đã ghi vào USD (physxJoint:maxJointVelocity), đặt trùng ở đây
            # để actuator không cố lệnh vượt qua mức PhysX sẽ cắt
            velocity_limit_sim=20.0,
            stiffness=0.0,
            # damping là lực cản tỉ lệ vận tốc, ăn thẳng vào lực điều khiển: ở 5 m/s thì
            # damping 0.5 đã nuốt 2.5 N. Để 0.02 cho gần như không cản mà drive vẫn không
            # hoàn toàn buông.
            damping=0.02,
        ),
        "pole": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_.*"],
            # khớp con lắc thụ động. Stiffness để 1e-5 chứ không phải 0: đủ nhỏ để không ảnh
            # hưởng động lực học (mô-men cỡ 1e-5 N·m ở lệch 1 rad) nhưng giữ cho PhysX coi bậc
            # tự do này là có drive, tránh trạng thái khớp thả hoàn toàn. effort_limit phải > 0
            # thì stiffness đó mới có tác dụng.
            effort_limit_sim=100.0,
            velocity_limit_sim=15.0,
            stiffness=1.0e-5,
            damping=0.0,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # mặc định = tư thế ĐÍCH (dựng đứng); reset lúc chạy mới quyết định bắt đầu ở đâu
        joint_pos={"Slider_1": 0.0, "Revolute_1": math.pi},
        joint_vel={"Slider_1": 0.0, "Revolute_1": 0.0},
    ),
)
