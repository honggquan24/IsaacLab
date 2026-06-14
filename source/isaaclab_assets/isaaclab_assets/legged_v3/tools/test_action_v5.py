"""Test action cho robot V5 — chạy các phase kiểm tra tuần tự.

Phase 0 (200 steps): zero action — quan sát robot thả tự do
Phase 1 (300 steps): hip sweep sin ±20° — kiểm tra 5-bar linkage
Phase 2 (300 steps): wheel forward — kiểm tra balance + di chuyển
Phase 3 (300 steps): wheel backward
Phase 4 (400 steps): kết hợp hip + wheel

Run:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/test_action_v5.py
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/test_action_v5.py --headless
"""

import math
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--fix_base", action="store_true", help="Treo robot cố định ở h=1m để test joint/wheel")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_assets.legged_v3.locomotion.legged_v5_wheel_env_cfg import LeggedV5WheelEnvCfg

# ──────────────────────────── Config ─────────────────────────────────────────

SUSPEND_ROBOT = args.fix_base

env_cfg = LeggedV5WheelEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device     = "cuda:0"

env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

robot      = env.scene["robot"]
jnames     = robot.data.joint_names
action_dim = env.action_manager.total_action_dim

print("\n" + "=" * 72)
print(f"V5 Robot — {robot.num_joints} joints / {robot.num_bodies} bodies")
print(f"Joints     : {jnames}")
print(f"Bodies     : {robot.data.body_names}")
print(f"Action dim : {action_dim}")
print("=" * 72 + "\n")

def jid(name):
    return jnames.index(name) if name in jnames else None

# Joint index map
HIP_R    = jid("right_hip_joint")
HIP_R_M  = jid("right_hip_joint_mimic")
HIP_L    = jid("left_hip_joint")
HIP_L_M  = jid("left_hip_joint_mimic")
KNEE_R1  = jid("right_knee_joint_1")
KNEE_R2  = jid("right_knee_joint_2")
KNEE_L1  = jid("left_knee_joint_1")
KNEE_L2  = jid("left_knee_joint_2")
WHEEL_R  = jid("right_wheel_joint")
WHEEL_L  = jid("left_wheel_joint")

# Action index trong action vector (MIMIC: chỉ 2 hip active, mimic tự = -active)
# HipMimicPositionActionCfg: right_hip, left_hip
# JointVelocityActionCfg:    right_wheel, left_wheel
ACT_HIP_R   = 0
ACT_HIP_L   = 1
ACT_WHL_R   = 2
ACT_WHL_L   = 3

# ──────────────────────────── Phase definition ───────────────────────────────

PHASES = [
    {"name": "ZERO      — thả tự do",          "steps": 200},
    {"name": "HIP SWEEP — sin ±20°",            "steps": 300},
    {"name": "WHEEL FWD — tiến thẳng",          "steps": 300},
    {"name": "WHEEL BWD — lùi thẳng",           "steps": 300},
    {"name": "COMBINED  — hip+wheel kết hợp",   "steps": 400},
]

DEV = env.device

# Vị trí treo cố định: h=1m, quay giống init_state
_SUSPEND_POS = torch.tensor([[0.0, 0.0, 1.0]], device=DEV)
_SUSPEND_ROT = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]], device=DEV)
_SUSPEND_VEL = torch.zeros(1, 6, device=DEV)

def suspend_robot():
    """Ghi đè root pose/vel về vị trí cố định, joint vẫn chạy tự do."""
    pose = torch.cat([_SUSPEND_POS, _SUSPEND_ROT], dim=-1)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(_SUSPEND_VEL)

def build_action(phase_step: int, phase: int) -> torch.Tensor:
    act = torch.zeros(1, action_dim, device=DEV)
    t   = phase_step * env_cfg.sim.dt * env_cfg.decimation  # thời gian thực trong phase (giây)

    if phase == 1:  # hip sin (mimic tự mirror, chỉ ra 2 lệnh active)
        amp = math.radians(20)
        val = amp * math.sin(2 * math.pi * 0.5 * t)       # 0.5 Hz
        act[0, ACT_HIP_R] =  val
        act[0, ACT_HIP_L] = -val

    elif phase == 2:  # wheel forward
        act[0, ACT_WHL_R] = 1.0
        act[0, ACT_WHL_L] = 1.0

    elif phase == 3:  # wheel backward
        act[0, ACT_WHL_R] = -1.0
        act[0, ACT_WHL_L] = -1.0

    elif phase == 4:  # combined
        amp = math.radians(15)
        val = amp * math.sin(2 * math.pi * 0.3 * t)
        act[0, ACT_HIP_R] =  val
        act[0, ACT_HIP_L] = -val
        act[0, ACT_WHL_R] =  0.5
        act[0, ACT_WHL_L] =  0.5

    return act

# ──────────────────────────── Run ────────────────────────────────────────────

def get_rpy_deg():
    q = robot.data.root_quat_w[0]          # w, x, y, z
    w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
    roll  = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2*(w*y - z*x)))))
    yaw   = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw

phase       = 0
phase_step  = 0   # bước trong phase hiện tại (tăng liên tục, KHÔNG reset khi episode kết thúc)
ep_step     = 0   # bước trong episode (reset khi terminate)
global_step = 0

print(f"\n▶ Phase {phase}: {PHASES[phase]['name']}")

def diag_wheel():
    """In trạng thái wheel joint để debug tại sao không quay."""
    wR, wL = WHEEL_R, WHEEL_L
    jvt = robot.data.joint_vel_target[0]   # velocity target đang set vào PhysX
    jv  = robot.data.joint_vel[0]          # velocity thực tế trong sim
    jks = robot.data.joint_stiffness[0]    # stiffness PhysX đang dùng
    jkd = robot.data.joint_damping[0]      # damping PhysX đang dùng
    jef = robot.data.applied_torque[0]     # torque ước tính (với ImplicitActuator là approx)
    print(
        f"  [DIAG WHEEL]"
        f"  vel_target R/L={jvt[wR]:+.2f}/{jvt[wL]:+.2f} r/s"
        f"  vel_actual R/L={jv[wR]:+.2f}/{jv[wL]:+.2f} r/s"
        f"  stiffness={jks[wR]:.1f}/{jks[wL]:.1f}"
        f"  damping={jkd[wR]:.1f}/{jkd[wL]:.1f}"
        f"  torque_est={jef[wR]:+.1f}/{jef[wL]:+.1f} Nm"
    )

while simulation_app.is_running():

    act = build_action(phase_step, phase)
    obs, reward, terminated, truncated, info = env.step(act)

    if SUSPEND_ROBOT:
        suspend_robot()

    phase_step  += 1
    ep_step     += 1
    global_step += 1

    # Log mỗi 20 steps
    if phase_step % 20 == 0:
        diag_wheel()
        h  = robot.data.root_pos_w[0, 2].item()
        roll, pitch, yaw = get_rpy_deg()

        def jp(idx):
            return f"{robot.data.joint_pos[0, idx].item():+.3f}" if idx is not None else "N/A"
        def jv(idx):
            return f"{robot.data.joint_vel[0, idx].item():+.2f}" if idx is not None else "N/A"
        def jt(idx):
            return f"{robot.data.applied_torque[0, idx].item():+.1f}" if idx is not None else "N/A"

        # In action hiện tại để confirm thay đổi
        act_str = " ".join(f"{act[0, i].item():+.2f}" for i in range(action_dim))

        print(
            f"[P{phase} ps{phase_step:4d} es{ep_step:3d}] "
            f"h={h:.3f}m  R={roll:+.1f}° P={pitch:+.1f}° Y={yaw:+.1f}° | "
            f"hip_R={jp(HIP_R)}({jt(HIP_R)}Nm)  hip_L={jp(HIP_L)}({jt(HIP_L)}Nm)  "
            f"whl={jv(WHEEL_R)}/{jv(WHEEL_L)}r/s | "
            f"act=[{act_str}] rew={reward[0]:.3f}"
        )

    # Reset episode nếu terminate — KHÔNG reset phase_step
    if terminated[0] or truncated[0]:
        term_mgr = env.termination_manager
        reasons  = [n for n in term_mgr.active_terms if term_mgr.get_term(n)[0]]
        print(f"  [ep reset at ep_step {ep_step} — {reasons}]")
        obs, _ = env.reset()
        if SUSPEND_ROBOT:
            suspend_robot()
        ep_step = 0

    # Chuyển phase dựa trên phase_step (không phụ thuộc episode)
    if phase_step >= PHASES[phase]["steps"]:
        phase      += 1
        phase_step  = 0
        ep_step     = 0
        obs, _      = env.reset()
        if SUSPEND_ROBOT:
            suspend_robot()

        if phase >= len(PHASES):
            print("\n✓ Hoàn thành tất cả phases.")
            break

        print(f"\n▶ Phase {phase}: {PHASES[phase]['name']}")

env.close()
simulation_app.close()
