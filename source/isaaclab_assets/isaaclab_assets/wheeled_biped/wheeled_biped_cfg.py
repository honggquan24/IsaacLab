# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ArticulationCfg cho robot bipedal wheel (V5) — load trực tiếp từ USD (Onshape export).

Cấu trúc 5-bar (Ascento/Diablo style):
  base
  ├── right_hip_joint        [MOTOR — crank A]
  ├── right_hip_joint_mimic  [MIMIC — crank B, mirror bằng HipMimicPositionAction]
  ├── right_knee_joint_1/2   [passive coupler]
  ├── right_wheel_joint      [MOTOR]
  └── (tương tự cho left)

Mimic KHÔNG dùng PhysxMimicJointAPI (không reliable qua session layer).
Thay vào đó dùng HipMimicPositionAction trong env cfg để tự ghi target = -hip_active.

NOTE contact sensors:
  USD export từ Onshape không có RigidBodyAPI trên child links — activate_contact_sensors
  không hoạt động được. Giải pháp đúng: convert từ URDF dùng Isaac Lab's convert_urdf.py.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

WHEELED_BIPED_USD_PATH = os.path.join(
    CURRENT_DIR,
    "usd",
    "wheeled_biped_fixed.usd",  # bánh xe đã gỡ khỏi vòng kín (fix_loop_v5.py)
)


WHEELED_BIPED_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=WHEELED_BIPED_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,
    ),
    soft_joint_pos_limit_factor=0.95,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.7071, 0.7071, 0.0, 0.0),  # 90° X — robot đứng thẳng vật lý
        joint_pos={
            "right_hip_joint": +0.0,
            "right_hip_joint_mimic": -0.0,
            "right_wheel_joint": 0.0,
            "left_hip_joint": -0.0,
            "left_hip_joint_mimic": +0.0,
            "left_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_active": IdealPDActuatorCfg(
            joint_names_expr=["right_hip_joint", "left_hip_joint"],
            effort_limit_sim=100.0,
            stiffness=30.0,
            damping=0.0,  # 1.5 — tăng nhẹ khử rung, vẫn dưới ngưỡng ~2 gây nảy (explicit PD)
            velocity_limit_sim=50.0,
        ),
        "hip_mimic": IdealPDActuatorCfg(
            joint_names_expr=["right_hip_joint_mimic", "left_hip_joint_mimic"],
            effort_limit_sim=100.0,
            stiffness=30.0,
            damping=0.0,  # 1.5 — tăng nhẹ khử rung, vẫn dưới ngưỡng ~2 gây nảy (explicit PD)
            velocity_limit_sim=50.0,
        ),
        "knee_passive": IdealPDActuatorCfg(
            joint_names_expr=[
                "right_knee_joint_1",
                "right_knee_joint_2",
                "left_knee_joint_1",
                "left_knee_joint_2",
            ],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
            velocity_limit_sim=50.0,
        ),
        # Bánh xe: ImplicitActuatorCfg (velocity-drive) — PhysX giải mô-men
        # = damping*(ω_target - ω). ĐÃ thử IdealPD (explicit) nhưng KHÔNG dùng được
        # trên joint bánh (không sinh mô-men) → buộc dùng Implicit (đã verify).
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint", "left_wheel_joint"],
            effort_limit_sim=500.0,
            stiffness=0.0,
            damping=150.0,
            velocity_limit_sim=500.0,
        ),
        "close_loop": ImplicitActuatorCfg(
            joint_names_expr=["right_close_loop_linear", "left_close_loop_linear"],
            effort_limit_sim=10000.0,
            stiffness=10000.0,
            damping=500.0,
            velocity_limit_sim=0.5,
        ),
    },
)
