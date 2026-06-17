"""Spawn test cho robot V5 — load USD trực tiếp, không cần env cfg.

Kiểm tra:
  - USD load được không
  - Tên joint / body đúng không
  - Robot đứng được (không sụp, không văng)
  - Wheel velocity khi cmd = 0

Run:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/spawn_robot_v5.py
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/spawn_robot_v5.py --headless
"""

from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── imports sau khi sim khởi động ─────────────────────────────────────────────
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.legged_v5.legged_v5_cfg import LEGGED_V5_CFG


def main():
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 200.0, device="cuda:0")
    )
    sim.set_camera_view(eye=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.3))

    # Ground plane
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    # Spawn robot
    robot_cfg = LEGGED_V5_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    # ── In thông tin joints / bodies ──────────────────────────────────────────
    joint_names = robot.data.joint_names
    body_names  = robot.data.body_names
    print("\n" + "=" * 70)
    print(f"V5 Robot — {robot.num_joints} joints, {robot.num_bodies} bodies")
    print(f"Joints : {joint_names}")
    print(f"Bodies : {body_names}")
    print("=" * 70 + "\n")

    def _ids(names):
        return [joint_names.index(n) for n in names if n in joint_names]

    hip_ids   = _ids(["right_hip_joint",       "left_hip_joint"])
    mimic_ids = _ids(["right_hip_joint_mimic",  "left_hip_joint_mimic"])
    knee_ids  = _ids(["right_knee_joint_1", "right_knee_joint_2",
                       "left_knee_joint_1",  "left_knee_joint_2"])
    wheel_ids = _ids(["right_wheel_joint",  "left_wheel_joint"])

    print(f"hip={hip_ids}  mimic={mimic_ids}  knee={knee_ids}  wheel={wheel_ids}\n")

    # Zero action
    zero_action = torch.zeros(1, robot.num_joints, device="cuda:0")

    step = 0
    while simulation_app.is_running():
        robot.set_joint_effort_target(zero_action)
        sim.step()
        robot.update(sim.cfg.dt)
        step += 1

        if step % 50 == 0:
            pos   = robot.data.root_pos_w[0]
            quat  = robot.data.root_quat_w[0]
            up_z  = (1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)).item()
            tilt  = torch.acos(torch.tensor(up_z).clamp(-1.0, 1.0)).item() * 57.296

            hip_pos   = robot.data.joint_pos[0, hip_ids].tolist()   if hip_ids   else []
            knee_pos  = robot.data.joint_pos[0, knee_ids].tolist()  if knee_ids  else []
            wheel_vel = robot.data.joint_vel[0, wheel_ids].tolist() if wheel_ids else []

            t = step * sim.cfg.dt
            print(f"[t={t:.2f}s step={step:5d}] "
                  f"h={pos[2]:.3f}m  tilt={tilt:.1f}°  "
                  f"hip={[f'{v:.3f}' for v in hip_pos]}  "
                  f"knee={[f'{v:.3f}' for v in knee_pos]}  "
                  f"wheel_vel={[f'{v:.2f}' for v in wheel_vel]}")

    sim.clear_all_callbacks()
    simulation_app.close()


if __name__ == "__main__":
    main()
