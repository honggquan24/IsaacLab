"""Soi mass/inertia RUNTIME + thử áp effort thuần lên bánh xe (bỏ qua actuator).

Mục tiêu: xác định bánh xe không quay là do (a) quán tính khổng lồ hay
(b) khớp bị ràng buộc cứng. Áp torque lớn trực tiếp và xem phản ứng.

    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/inspect_runtime_v5.py --headless
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_assets.legged_v3.locomotion.legged_v5_wheel_env_cfg import LeggedV5WheelEnvCfg

cfg = LeggedV5WheelEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
robot = env.scene["robot"]
DEV = env.device

jn = robot.data.joint_names
bn = robot.data.body_names
print("\nJOINTS:", jn)
print("BODIES:", bn)

# ── Mass / inertia runtime ───────────────────────────────────────────────────
masses = robot.root_physx_view.get_masses()[0]
inertias = robot.root_physx_view.get_inertias()[0]   # shape (num_bodies, 9)
print("\n── MASS / INERTIA (runtime) ──")
for i, name in enumerate(bn):
    inr = inertias[i].reshape(3, 3)
    diag = (inr[0, 0].item(), inr[1, 1].item(), inr[2, 2].item())
    print(f"  {name:<24} mass={masses[i].item():.5f} kg   Idiag=({diag[0]:.3e}, {diag[1]:.3e}, {diag[2]:.3e})")

# ── Áp effort thuần lên bánh, treo robot, hip giữ 0 ──────────────────────────
wR = jn.index("right_wheel_joint")
wL = jn.index("left_wheel_joint")

SUS_POSE = torch.tensor([[0., 0., 1., 0.7071, 0.7071, 0., 0.]], device=DEV)
SUS_VEL = torch.zeros(1, 6, device=DEV)

SIM_DT = env.cfg.sim.dt
JIDX = torch.tensor([wR, wL], device=DEV)

def manual_loop(label, setup):
    print(f"\n── {label} ──")
    # reset velocity về 0
    jv = robot.data.joint_vel.clone(); jv[:] = 0.0
    robot.write_joint_state_to_sim(robot.data.joint_pos, jv)
    for step in range(80):
        setup()
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(SIM_DT)
        robot.write_root_pose_to_sim(SUS_POSE)
        robot.write_root_velocity_to_sim(SUS_VEL)
        if step % 20 == 0:
            v = robot.data.joint_vel[0]
            ks = robot.data.joint_stiffness[0]; kd = robot.data.joint_damping[0]
            print(f"  step {step:3d}  wheelvel R/L = {v[wR].item():+.2f}/{v[wL].item():+.2f}  "
                  f"k={ks[wR].item():.1f} d={kd[wR].item():.1f}")

def setup_effort():
    eff = torch.zeros(1, robot.num_joints, device=DEV)
    eff[0, wR] = 50.0; eff[0, wL] = 50.0
    robot.set_joint_effort_target(eff)

def setup_veltarget():
    tgt = torch.full((1, 2), 10.0, device=DEV)
    robot.set_joint_velocity_target(tgt, joint_ids=JIDX)

def setup_physx_veltarget():
    full = robot.root_physx_view.get_dof_velocity_targets().clone()
    full[0, wR] = 10.0; full[0, wL] = 10.0
    robot.root_physx_view.set_dof_velocity_targets(full, torch.tensor([0], device=DEV))

manual_loop("A) EFFORT thuần 50 N·m (vòng sim thủ công)", setup_effort)
manual_loop("B) set_joint_velocity_target(+10) — Articulation API", setup_veltarget)
manual_loop("C) set_dof_velocity_targets(+10) — PhysX view thẳng", setup_physx_veltarget)

app.close()
