"""Test động cơ V5 — kiểm tra hip (position) và wheel (velocity).

Modes:
  --mode hip    : sine wave cho hip_joint phải + trái, wheel = 0
  --mode wheel  : step velocity cho wheel, hip = 0
  --mode all    : cả hai đồng thời

--fix_base : treo robot cố định trong không khí (fixed joint tới world)
             dùng khi muốn test motor độc lập, không bị ảnh hưởng bởi chuyển động robot

Run:
    ./isaaclab.sh -p .../spawn_test_motor_v5.py --fix_base --mode wheel --cmd 3.0
    ./isaaclab.sh -p .../spawn_test_motor_v5.py --fix_base --mode hip --amp 0.3
    ./isaaclab.sh -p .../spawn_test_motor_v5.py --mode all
    ./isaaclab.sh -p .../spawn_test_motor_v5.py --fix_base --headless --mode wheel
"""

from isaaclab.app import AppLauncher
import argparse
import math

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--mode",      default="all",   choices=["hip", "wheel", "all"])
parser.add_argument("--cmd",       type=float, default=3.0,   help="Wheel velocity cmd (rad/s)")
parser.add_argument("--amp",       type=float, default=0.3,   help="Hip sine amplitude (rad)")
parser.add_argument("--freq",      type=float, default=0.5,   help="Hip sine frequency (Hz)")
parser.add_argument("--fix_base",  action="store_true",       help="Treo robot cố định trong không khí")
parser.add_argument("--base_link", default="base",            help="Tên base link trong USD (default: base)")
parser.add_argument("--spawn_z",   type=float, default=1.0,   help="Chiều cao spawn khi fix_base (m)")
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.legged_v3.legged_v5_cfg import LEGGED_V5_CFG

SIM_DT   = 1 / 200.0
PRINT_HZ = 10
SEP      = "=" * 72


def _fix_base_to_world(robot_prim_path: str, base_link_name: str, z: float):
    """Tạo FixedJoint giữa world và base link — treo robot tại chỗ."""
    import omni.usd
    from pxr import UsdPhysics, Sdf
    stage = omni.usd.get_context().get_stage()

    # Thử tìm base link theo các pattern thường gặp
    candidates = [
        f"{robot_prim_path}/{base_link_name}",
        f"{robot_prim_path}/base_link",
        f"{robot_prim_path}/base",
    ]
    base_path = None
    for c in candidates:
        if stage.GetPrimAtPath(c).IsValid():
            base_path = c
            break

    if base_path is None:
        # In ra tất cả child prims để debug
        root = stage.GetPrimAtPath(robot_prim_path)
        children = [str(p.GetPath()) for p in root.GetChildren()] if root.IsValid() else []
        print(f"[fix_base] WARNING: base link không tìm thấy. Children: {children}")
        return

    fj = UsdPhysics.FixedJoint.Define(stage, "/World/RobotFixBase")
    fj.CreateBody0Rel().SetTargets([])                    # body0 = world frame
    fj.CreateBody1Rel().SetTargets([Sdf.Path(base_path)]) # body1 = robot base
    print(f"[fix_base] FixedJoint: world → {base_path}  (z={z:.2f}m)")


def main():
    spawn_z = args.spawn_z if args.fix_base else 0.30

    sim = SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0"))
    sim.set_camera_view(eye=(2.0, 1.5, 1.5), target=(0.0, 0.0, spawn_z))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    robot_cfg = LEGGED_V5_CFG.replace(
        prim_path="/World/Robot",
        init_state=LEGGED_V5_CFG.init_state.replace(pos=(0.0, 0.0, spawn_z)),
    )
    robot = Articulation(robot_cfg)

    # Treo robot trước khi sim.reset() khởi động physics
    if args.fix_base:
        _fix_base_to_world("/World/Robot", args.base_link, spawn_z)

    sim.reset()

    jnames = robot.data.joint_names
    print(f"\n{SEP}")
    print(f"V5 Motor Test — mode={args.mode}  fix_base={args.fix_base}")
    print(f"Joints ({robot.num_joints}): {jnames}")
    print(SEP + "\n")

    def ids(*names):
        return [jnames.index(n) for n in names if n in jnames]

    hip_ids   = ids("right_hip_joint",   "left_hip_joint")
    wheel_ids = ids("right_wheel_joint", "left_wheel_joint")
    knee_ids  = ids("right_knee_joint_1", "right_knee_joint_2",
                    "left_knee_joint_1",  "left_knee_joint_2")

    if not hip_ids:
        print("[WARN] Không tìm thấy hip joints — kiểm tra tên joint trong USD")
    if not wheel_ids:
        print("[WARN] Không tìm thấy wheel joints — kiểm tra tên joint trong USD")

    hip_target   = torch.zeros(1, len(hip_ids),   device="cuda:0")
    wheel_target = torch.zeros(1, len(wheel_ids), device="cuda:0")

    print(f"{'t(s)':>6} | {'h(m)':>6} {'tilt°':>6} | "
          f"{'hip_R':>7} {'hip_L':>7} | "
          f"{'wvel_R':>7} {'wvel_L':>7} | "
          f"{'knee0':>7} {'knee1':>7}")
    print("-" * 72)

    step = 0
    print_every = max(1, int(1.0 / (SIM_DT * PRINT_HZ)))

    while simulation_app.is_running():
        t = step * SIM_DT

        if args.mode in ("hip", "all"):
            hip_target[:] = args.amp * math.sin(2 * math.pi * args.freq * t)
        else:
            hip_target[:] = 0.0

        if args.mode in ("wheel", "all"):
            wheel_target[:] = args.cmd
        else:
            wheel_target[:] = 0.0

        if hip_ids:
            robot.set_joint_position_target(
                hip_target, joint_ids=torch.tensor(hip_ids, device="cuda:0"))
        if wheel_ids:
            robot.set_joint_velocity_target(
                wheel_target, joint_ids=torch.tensor(wheel_ids, device="cuda:0"))

        sim.step()
        robot.update(SIM_DT)
        step += 1

        if step % print_every == 0:
            pos  = robot.data.root_pos_w[0]
            quat = robot.data.root_quat_w[0]
            up_z = (1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)).clamp(-1.0, 1.0)
            tilt = torch.acos(up_z).item() * 57.296

            hip_pos  = robot.data.joint_pos[0, hip_ids].tolist()  if hip_ids   else [0.0, 0.0]
            wvel     = robot.data.joint_vel[0, wheel_ids].tolist() if wheel_ids else [0.0, 0.0]
            knee_pos = robot.data.joint_pos[0, knee_ids].tolist()  if knee_ids  else [0.0, 0.0]

            print(f"{t:6.2f} | {pos[2].item():6.3f} {tilt:6.1f} | "
                  f"{hip_pos[0] if len(hip_pos)>0 else 0:7.3f} "
                  f"{hip_pos[1] if len(hip_pos)>1 else 0:7.3f} | "
                  f"{wvel[0] if len(wvel)>0 else 0:7.2f} "
                  f"{wvel[1] if len(wvel)>1 else 0:7.2f} | "
                  f"{knee_pos[0] if len(knee_pos)>0 else 0:7.3f} "
                  f"{knee_pos[1] if len(knee_pos)>1 else 0:7.3f}")

    sim.clear_all_callbacks()
    simulation_app.close()


if __name__ == "__main__":
    main()
