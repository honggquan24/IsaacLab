"""Custom event functions cho Legged Robot V3/V5."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_hip_mimic_joint_api(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Apply PhysxMimicJointAPI (gearing=-1.0) cho hip_mimic joints.

    Cần chạy 1 lần lúc startup. MimicJointAPI nằm trong articulation nên
    hoạt động đúng trên mọi env sau khi clone — khác với excludeFromArticulation.

    Pair:
        right_hip_joint_mimic  ←  gearing=-1.0  ←  right_hip_joint
        left_hip_joint_mimic   ←  gearing=-1.0  ←  left_hip_joint
    """
    import omni.usd
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()

    PAIRS = [
        ("right_hip_joint_mimic", "right_hip_joint"),
        ("left_hip_joint_mimic",  "left_hip_joint"),
    ]

    for env_id in env_ids.tolist():
        base = f"/World/envs/env_{env_id}/Robot/Robot/Robot"
        for slave_name, master_name in PAIRS:
            slave  = stage.GetPrimAtPath(f"{base}/{slave_name}")
            master = stage.GetPrimAtPath(f"{base}/{master_name}")
            if not slave.IsValid() or not master.IsValid():
                continue
            api = PhysxSchema.PhysxMimicJointAPI.Apply(slave, "rotX")
            api.GetGearingAttr().Set(-1.0)
            api.GetOffsetAttr().Set(0.0)
            api.GetReferenceJointRel().SetTargets([master.GetPath()])
