# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ArticulationCfg cho con lắc kép trên xe đẩy (CAD tự dựng).

Env mặc định (:mod:`cart_pendulum_double_env_cfg`) dùng ``CART_DOUBLE_PENDULUM_CFG``
của Isaac Lab (tải từ nucleus) để so sánh với baseline. Cfg dưới đây trỏ vào bản
CAD trong ``usd/`` — gán vào ``scene.robot`` nếu muốn chạy mô hình thật:

    from .cart_pendulum_double_cfg import CART_PENDULUM_DOUBLE_CFG

    robot: Articulation = CART_PENDULUM_DOUBLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CART_PENDULUM_DOUBLE_USD_PATH = os.path.join(CURRENT_DIR, "usd", "cart_pendulum_double_cfg.usd")

CART_PENDULUM_DOUBLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CART_PENDULUM_DOUBLE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            linear_damping=0.002,
            angular_damping=0.005,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            max_contact_impulse=500,
            retain_accelerations=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={"Revolute_1": math.pi},
        joint_vel={"Slider_1": 0.1},
    ),
    actuators={
        # Xe đẩy: truyền động duy nhất, PD tường minh (stiffness=0) → lệnh là lực.
        "slider": ImplicitActuatorCfg(
            joint_names_expr=["Slider_1"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
            velocity_limit_sim=10.0,
        ),
        # Hai khớp con lắc thụ động, chỉ để ma sát nhớt rất nhỏ.
        "pendulum_1": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],
            effort_limit_sim=400.0,
            stiffness=0.001,
            damping=0.01,
            velocity_limit_sim=20.0,
        ),
        "pendulum_2": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_2"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
            velocity_limit_sim=25.0,
        ),
    },
)
