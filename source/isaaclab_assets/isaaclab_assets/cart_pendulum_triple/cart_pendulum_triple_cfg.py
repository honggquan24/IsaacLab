# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cấu hình robot con lắc ba trên xe đẩy (CAD Onshape).

Cùng ray và cùng xe với con lắc đơn/kép, ba khâu nối tiếp nhau. Quy ước góc giữ nguyên:
``Revolute_1`` bằng 0 là khâu 1 thõng xuống (dựng đứng là π), còn ``Revolute_2`` và
``Revolute_3`` bằng 0 nghĩa là thẳng hàng với khâu trước. Mọi khớp bằng 0 tuyệt đối là cả
chuỗi thõng thẳng xuống; vị trí khớp mặc định dưới đây là cả chuỗi dựng thẳng đứng.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CART_PENDULUM_TRIPLE_USD_PATH = os.path.join(CURRENT_DIR, "usd", "cart_pendulum_triple_cfg.usd")
"""USD đã vá. Bản Onshape thô nằm cạnh nó ở ``cart_pendulum_triple_base.usd``."""

CART_PENDULUM_TRIPLE_RAIL_LIMIT = 0.555
"""Nửa chiều dài ray [m]. Sửa lại nếu ray trong CAD của bạn khác con lắc đơn."""

CART_PENDULUM_TRIPLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CART_PENDULUM_TRIPLE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # chặn thêm ở mức thân, cao hơn trần khớp một chút để trần khớp mới là cái ràng buộc
            max_linear_velocity=30.0,
            max_angular_velocity=30.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # ba khâu nối tiếp là chuỗi dài và nhẹ, thiếu vòng lặp solver là khớp giãn ra khi lắc
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=2,
        ),
    ),
    actuators={
        "cart": ImplicitActuatorCfg(
            joint_names_expr=["Slider_1"],
            effort_limit_sim=20.0,
            # trần khớp cũng đã ghi vào USD (physxJoint:maxJointVelocity), đặt trùng ở đây
            # để actuator không cố lệnh vượt qua mức PhysX sẽ cắt
            velocity_limit_sim=12.0,
            stiffness=0.0,
            # damping là lực cản tỉ lệ vận tốc, ăn thẳng vào lực điều khiển: ở 5 m/s thì
            # damping 0.5 đã nuốt 2.5 N. Để 0.05 cho gần như không cản mà drive vẫn không
            # hoàn toàn không tắt dần.
            damping=0.05,
        ),
        "pole": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_.*"],
            # xem chú thích trong cart_pendulum_cfg.py
            effort_limit_sim=1.0,
            velocity_limit_sim=15.0,
            stiffness=1.0e-5,
            damping=0.0,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # mặc định = tư thế ĐÍCH (cả chuỗi dựng đứng); reset lúc chạy mới quyết định bắt đầu ở đâu
        joint_pos={"Slider_1": 0.0, "Revolute_1": math.pi, "Revolute_2": 0.0, "Revolute_3": 0.0},
        joint_vel={"Slider_1": 0.0, "Revolute_1": 0.0, "Revolute_2": 0.0, "Revolute_3": 0.0},
    ),
)
