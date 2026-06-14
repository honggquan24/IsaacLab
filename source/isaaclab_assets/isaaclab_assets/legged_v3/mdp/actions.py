"""Custom actions cho V5: hip với auto-mimic.

HipMimicPositionAction:
  - Input (policy): 2 giá trị cho right_hip_joint và left_hip_joint
  - Output (sim): 4 targets — hip_active giữ nguyên, hip_mimic = -hip_active
  - Không dùng PhysxMimicJointAPI (không reliable qua session layer)
"""
from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HipMimicPositionAction(ActionTerm):
    """Position action cho hip_active, tự mirror sang hip_mimic (gearing=-1)."""

    cfg: HipMimicPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: HipMimicPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Tìm joint indices
        all_joints = self._asset.data.joint_names
        self._active_ids = [all_joints.index(n) for n in cfg.active_joint_names]
        self._mimic_ids  = [all_joints.index(n) for n in cfg.mimic_joint_names]

        if len(self._active_ids) != len(self._mimic_ids):
            raise ValueError("active_joint_names và mimic_joint_names phải có cùng số lượng.")

        self._num_active = len(self._active_ids)
        self._raw_actions = torch.zeros(env.num_envs, self._num_active, device=env.device)
        self._processed   = torch.zeros(env.num_envs, self._num_active, device=env.device)

    @property
    def action_dim(self) -> int:
        return self._num_active

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed = actions * self.cfg.scale

    def apply_actions(self):
        targets = self._processed
        self._asset.set_joint_position_target(targets, joint_ids=self._active_ids)
        self._asset.set_joint_position_target(-targets, joint_ids=self._mimic_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # reset_joints_by_offset đã cập nhật data.joint_pos cho env_ids.
        # Sync mimic = -active để 5-bar luôn đúng ràng buộc trước step đầu.
        if env_ids is None:
            active_pos = self._asset.data.joint_pos[:, self._active_ids]
            self._asset.write_joint_position_to_sim(-active_pos, joint_ids=self._mimic_ids)
            self._asset.set_joint_position_target(active_pos,  joint_ids=self._active_ids)
            self._asset.set_joint_position_target(-active_pos, joint_ids=self._mimic_ids)
            self._raw_actions.zero_()
            self._processed.zero_()
        else:
            active_pos = self._asset.data.joint_pos[env_ids][:, self._active_ids]
            self._asset.write_joint_position_to_sim(-active_pos, joint_ids=self._mimic_ids, env_ids=env_ids)
            self._asset.set_joint_position_target(active_pos,  joint_ids=self._active_ids, env_ids=env_ids)
            self._asset.set_joint_position_target(-active_pos, joint_ids=self._mimic_ids, env_ids=env_ids)
            self._raw_actions[env_ids] = 0.0
            self._processed[env_ids] = 0.0


@configclass
class HipMimicPositionActionCfg(ActionTermCfg):
    """Config cho HipMimicPositionAction."""

    class_type: type = HipMimicPositionAction

    asset_name: str = MISSING
    active_joint_names: list[str] = MISSING
    mimic_joint_names:  list[str] = MISSING
    scale: float = 1.0
