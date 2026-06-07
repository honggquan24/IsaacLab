"""Command terms cho Biped PID-RL cascade."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TargetTiltCommand(CommandTerm):
    """Sinh target tilt angle ngẫu nhiên (step input cho vòng trong).

    command shape (N, 3): [roll_des, pitch_des, yaw_des]
    yaw_des là góc tuyệt đối (world frame).
    """

    cfg: "TargetTiltCommandCfg"

    def __init__(self, cfg: "TargetTiltCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids):
        from isaaclab.utils.math import euler_xyz_from_quat
        n = len(env_ids)
        roll  = torch.empty(n, device=self.device).uniform_(*self.cfg.roll_range)
        pitch = torch.empty(n, device=self.device).uniform_(*self.cfg.pitch_range)
        yaw_delta = torch.empty(n, device=self.device).uniform_(*self.cfg.yaw_delta_range)
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w[env_ids])
        self._command[env_ids, 0] = roll
        self._command[env_ids, 1] = pitch
        self._command[env_ids, 2] = yaw + yaw_delta

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass


@configclass
class TargetTiltCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = TargetTiltCommand
    asset_name:            str   = "robot"
    resampling_time_range: tuple[float, float] = (3.0, 6.0)
    roll_range:            tuple[float, float] = (-0.25, 0.25)   # rad ~8.6°
    pitch_range:           tuple[float, float] = (-0.25, 0.25)   # rad
    yaw_delta_range:       tuple[float, float] = (-1.57,  1.57)    # rad ~17°
