"""Diagnostic test: why does legged_v3 always fall?

Tests 4 scenarios to isolate the root cause:
  1. ZERO actions — does physics alone keep the robot upright?
  2. ZERO wheel, fixed hip — does PD hold hip position?
  3. MANUAL balance wheel — simple inverted-pendulum P-control on tilt
  4. Print contact forces, joint torques, loop closure geometry at spawn

Run:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/spawn_test_diagnostic.py
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/spawn_test_diagnostic.py --headless
"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--scenario", type=int, default=0,
    help="0=all, 1=zero_actions, 2=fixed_hip, 3=manual_balance")
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import math
from isaaclab_assets.legged_v3.locomotion.legged_v3_wheel_env_cfg import LeggedV3WheelEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

SEP = "=" * 70


def build_env(num_envs=1, lenient_termination=True):
    env_cfg = LeggedV3WheelEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = "cuda:0"
    env_cfg.viewer.eye    = (1.5, 1.5, 0.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.2)
    if lenient_termination:
        # Relax termination for diagnostic: let the robot fall further before reset
        # so we can observe actual fall dynamics, not just early safety cuts.
        if hasattr(env_cfg.terminations, "base_height_min"):
            env_cfg.terminations.base_height_min.params["minimum_height"] = 0.05
        if hasattr(env_cfg.terminations, "bad_orientation"):
            env_cfg.terminations.bad_orientation.params["limit_angle"] = math.pi * 2 / 3  # 120°
    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


def get_state(robot, env):
    """Return a dict with key diagnostic quantities."""
    pos   = robot.data.root_pos_w[0]        # (3,)
    quat  = robot.data.root_quat_w[0]       # (w,x,y,z)
    w, x, y, z = quat
    up_z  = (1.0 - 2.0 * (x*x + y*y)).item()
    tilt  = math.acos(max(-1.0, min(1.0, up_z))) * 57.296

    jnames = robot.data.joint_names
    jpos   = robot.data.joint_pos[0]
    jvel   = robot.data.joint_vel[0]
    jtorq  = robot.data.applied_torque[0]

    bnames = robot.data.body_names
    bpos_w = robot.data.body_pos_w[0]       # (num_bodies, 3)

    def jidx(name_pattern):
        return [i for i, n in enumerate(jnames) if name_pattern in n]

    def bidx(name_pattern):
        return [i for i, n in enumerate(bnames) if name_pattern in n]

    return dict(
        height=pos[2].item(),
        tilt_deg=tilt,
        up_z=up_z,
        jnames=jnames,
        jpos=jpos,
        jvel=jvel,
        jtorq=jtorq,
        bnames=bnames,
        bpos_w=bpos_w,
        hip_ids=jidx("hip_joint_A1"),
        wheel_ids=jidx("wheel_joint"),
        foot_ids=bidx("foot_link"),
    )


def print_state(s, step, label=""):
    foot_z = s["bpos_w"][s["foot_ids"], 2].tolist() if s["foot_ids"] else []
    hip_pos  = s["jpos"][s["hip_ids"]].tolist()
    hip_torq = s["jtorq"][s["hip_ids"]].tolist()
    wvel     = s["jvel"][s["wheel_ids"]].tolist()
    wtorq    = s["jtorq"][s["wheel_ids"]].tolist()

    print(f"  [{label} step {step:4d}] "
          f"h={s['height']:.3f}m  tilt={s['tilt_deg']:.1f}°  "
          f"foot_z={[f'{v:.3f}' for v in foot_z]}  "
          f"hip_pos={[f'{v:.3f}' for v in hip_pos]}  "
          f"hip_torq={[f'{v:.1f}' for v in hip_torq]}  "
          f"wvel={[f'{v:.2f}' for v in wvel]}  "
          f"wtorq={[f'{v:.2f}' for v in wtorq]}")


def print_spawn_geometry(robot):
    """Check body positions at spawn — are the wheels touching the ground?"""
    bnames = robot.data.body_names
    bpos   = robot.data.body_pos_w[0]
    print(f"\n{SEP}")
    print("SPAWN GEOMETRY (world Z of each body)")
    print(SEP)
    for i, name in enumerate(bnames):
        z = bpos[i, 2].item()
        marker = " ← foot" if "foot" in name else ""
        marker += " ← base" if name == "base_link" else ""
        print(f"  {name:35s}  z={z:.4f}m{marker}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: All-zero actions — pure physics
# ─────────────────────────────────────────────────────────────────────────────
def run_zero_actions(max_steps=200):
    print(f"\n{SEP}")
    print("SCENARIO 1: ZERO ACTIONS (pure physics / PD at zero)")
    print(SEP)
    print("Expected: robot stays upright if PD stiffness is sufficient and")
    print("          stance geometry is correct. Falls → physics problem.")
    env = build_env()
    obs, _ = env.reset()
    robot = env.scene["robot"]
    print_spawn_geometry(robot)

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros(1, action_dim, device="cuda:0")
    marker = make_imu_marker()

    episode = 0
    for step in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        update_imu_marker(marker, env)
        s = get_state(robot, env)
        if step % 20 == 0:
            print_state(s, step, "ZERO")
        if terminated[0] or truncated[0]:
            print(f"\n  *** [ep {episode}] TERMINATED at step {step} | tilt={s['tilt_deg']:.1f}° | height={s['height']:.3f}m ***\n")
            episode += 1
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Fixed hip (command=0), check if PD holds
# ─────────────────────────────────────────────────────────────────────────────
def run_fixed_hip(max_steps=200):
    print(f"\n{SEP}")
    print("SCENARIO 2: HIP COMMAND=0 (test PD hold), WHEEL=0")
    print(SEP)
    print("Expected: hip stays near 0 with hip_torque proportional to error.")
    env = build_env()
    obs, _ = env.reset()
    robot = env.scene["robot"]
    # action dim order: hip_left, hip_right, wheel_left, wheel_right
    action_dim = env.action_manager.total_action_dim
    action = torch.zeros(1, action_dim, device="cuda:0")

    marker = make_imu_marker()
    episode = 0
    for step in range(max_steps):
        s = get_state(robot, env)
        action[0, :2] = 0.0
        action[0, 2:] = 0.0
        obs, reward, terminated, truncated, info = env.step(action)
        update_imu_marker(marker, env)
        if step % 20 == 0:
            print_state(s, step, "HOLD")
        if terminated[0] or truncated[0]:
            print(f"\n  *** [ep {episode}] TERMINATED at step {step} | tilt={s['tilt_deg']:.1f}° ***\n")
            episode += 1
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Manual balance
# ─────────────────────────────────────────────────────────────────────────────
def run_manual_balance(max_steps=2000, kp_fwd=25.0, kd_fwd=0.3, kp_roll=3.0, kd_roll=0.2):
    """Balance dùng projected_gravity_b (body frame).

    projected_gravity_b khi thẳng đứng = (0, 0, -1).
    gx < 0 → robot đang ngã về phía trước (+X) → cần bánh tiến.
    gy ≠ 0 → robot đang ngã ngang (roll) → cần hip correction.

    Action order: [left_hip, right_hip, left_wheel, right_wheel]
    Scale: hip=1.0, wheel=3.0 rad/s per unit action.
    """
    print(f"\n{SEP}")
    print(f"SCENARIO 3: MANUAL BALANCE  kp_fwd={kp_fwd}  kd_fwd={kd_fwd}  kp_roll={kp_roll}  kd_roll={kd_roll}")
    print(SEP)
    env = build_env()
    obs, _ = env.reset()
    robot = env.scene["robot"]
    action_dim = env.action_manager.total_action_dim
    marker = make_imu_marker()

    prev_s = None
    for step in range(max_steps):
        # Projected gravity in robot body frame: upright = (0, 0, -1)
        g = robot.data.projected_gravity_b[0]   # (3,)
        gx, gy, gz = float(g[0]), float(g[1]), float(g[2])

        # Pitch = forward/backward tilt in robot body frame
        pitch      = math.atan2(-gx, math.sqrt(gy**2 + gz**2))
        roll       = math.atan2(gy, -gz)
        pitch_rate = float(robot.data.root_ang_vel_b[0, 1])   # ω_y

        # Wheel-only balance: hip stays at 0 (let hip PD hold natural stance).
        # Hip joints are fore-aft — cannot correct lateral roll directly.
        fwd_cmd = float(torch.clamp(
            torch.tensor(kp_fwd * pitch + kd_fwd * pitch_rate),
            -1.0, 1.0,
        ))

        action = torch.zeros(1, action_dim, device="cuda:0")
        # action[0, 0..1] = 0: hip holds target=0 via stiffness=80
        action[0, 2] = fwd_cmd   # left  wheel
        action[0, 3] = fwd_cmd   # right wheel

        s_before = get_state(robot, env)
        obs, reward, terminated, truncated, info = env.step(action)
        update_imu_marker(marker, env)
        s = get_state(robot, env)

        if step % 50 == 0:
            print_state(s_before, step, "BAL-PD")
            print(f"           pitch={math.degrees(pitch):+.1f}°  roll={math.degrees(roll):+.1f}°  "
                  f"pitch_rate={pitch_rate:+.3f}  fwd_cmd={fwd_cmd:+.3f}")
        if terminated[0] or truncated[0]:
            # Print state AT termination (before reset)
            foot_z = s_before["bpos_w"][s_before["foot_ids"], 2].tolist()
            print(f"\n  *** TERMINATED at step {step} "
                  f"| tilt={s_before['tilt_deg']:.1f}°  h={s_before['height']:.3f}m "
                  f"foot_z={[f'{v:.3f}' for v in foot_z]} "
                  f"pitch={math.degrees(pitch):+.1f}°  roll={math.degrees(roll):+.1f}° ***\n")
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Check contact forces at spawn
# ─────────────────────────────────────────────────────────────────────────────
def run_contact_check(max_steps=10):
    print(f"\n{SEP}")
    print("SCENARIO 4: CONTACT FORCES AT SPAWN")
    print(SEP)
    env = build_env()
    obs, _ = env.reset()
    robot = env.scene["robot"]

    # Check if contact sensor is available
    if "contact_forces_body" in env.scene.keys():
        sensor = env.scene["contact_forces_body"]
        bnames = sensor.body_names   # attribute on sensor, not sensor.data
        print(f"Contact sensor bodies ({len(bnames)}): {bnames}")
    else:
        print("No contact_forces_body sensor found!")
        bnames = []

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros(1, action_dim, device="cuda:0")

    for step in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if "contact_forces_body" in env.scene.keys():
            sensor = env.scene["contact_forces_body"]
            # net_forces_w: (num_envs, num_bodies, 3)
            forces = sensor.data.net_forces_w[0]        # (num_bodies, 3)
            force_mag = forces.norm(dim=-1)              # (num_bodies,)
            print(f"\n  [step {step}] Contact forces (N):")
            for i, (bname, fmag) in enumerate(zip(bnames, force_mag.tolist())):
                if fmag > 0.1:
                    print(f"    {bname:35s}  |F|={fmag:.2f}N  ← CONTACT")
        if terminated[0]:
            print(f"  *** TERMINATED at step {step} ***")
            break

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Wheel direction test — vx hay vy?
# ─────────────────────────────────────────────────────────────────────────────
def run_direction_test(steps_per_phase=100):
    """Command fixed wheel velocity, observe which world axis the robot moves along.

    Phases:
      A: both wheels +1  → should move in +X or +Y
      B: both wheels -1  → opposite direction
      C: left +1, right -1 → should spin (yaw rotation)
      D: left -1, right +1 → opposite spin
    """
    print(f"\n{SEP}")
    print("SCENARIO 5: DIRECTION TEST (vx hay vy?)")
    print(SEP)
    print("Quan sát: robot tiến theo trục nào khi cả 2 bánh cùng chiều?")
    env = build_env()
    obs, _ = env.reset()
    robot = env.scene["robot"]
    action_dim = env.action_manager.total_action_dim

    phases = [
        ("A: BOTH +1  (forward?)", [ 0.0,  0.0, +1.0, +1.0]),
        ("B: BOTH -1  (backward?)",[ 0.0,  0.0, -1.0, -1.0]),
        ("C: LEFT+1 RIGHT-1 (yaw+?)",[ 0.0, 0.0, +1.0, -1.0]),
        ("D: LEFT-1 RIGHT+1 (yaw-?)",[ 0.0, 0.0, -1.0, +1.0]),
    ]

    for phase_name, action_vals in phases:
        obs, _ = env.reset()
        start_pos = robot.data.root_pos_w[0].clone()
        action = torch.tensor([action_vals], dtype=torch.float32, device="cuda:0")

        print(f"\n  Phase {phase_name}")
        print(f"  Start pos: x={start_pos[0]:.3f}  y={start_pos[1]:.3f}  z={start_pos[2]:.3f}")

        for step in range(steps_per_phase):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated[0] or truncated[0]:
                print(f"    Terminated at step {step}")
                break

        pos = robot.data.root_pos_w[0]
        quat = robot.data.root_quat_w[0]
        w, x, y, z = quat
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        delta_x = (pos[0] - start_pos[0]).item()
        delta_y = (pos[1] - start_pos[1]).item()
        delta_yaw = math.degrees(yaw)

        print(f"  End   pos: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
        print(f"  Δx={delta_x:+.3f}m  Δy={delta_y:+.3f}m  yaw={delta_yaw:+.1f}°")
        dominant = "X" if abs(delta_x) > abs(delta_y) else "Y"
        print(f"  → Robot moves mainly along {dominant} axis")

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6: Reward check khi di chuyển theo X
# ─────────────────────────────────────────────────────────────────────────────
def run_reward_check(max_steps=300):
    """Manual balance + forward drive. Print từng reward term để xem reward có tăng không."""
    print(f"\n{SEP}")
    print("SCENARIO 6: REWARD CHECK — di chuyển theo X")
    print(SEP)
    print("Phase 1 (step 0-99):   đứng yên  (wheel=0)")
    print("Phase 2 (step 100-199): tiến  (wheel=+0.5)")
    print("Phase 3 (step 200-299): lùi   (wheel=-0.5)")
    print(SEP)

    env = build_env()
    obs, _ = env.reset()
    robot  = env.scene["robot"]
    action_dim = env.action_manager.total_action_dim

    # Lấy tên các reward terms
    reward_mgr = env.reward_manager
    prev_sums = None   # track episode_sums delta to get per-step reward

    for step in range(max_steps):
        # Balance PD (giống scenario 3)
        quat = robot.data.root_quat_w[0]
        w, x, y, z = quat
        pitch_angle = math.asin(max(-1.0, min(1.0, (2*(w*y - z*x)).item())))
        pitch_rate  = robot.data.root_ang_vel_b[0, 1].item()
        balance_cmd = float(torch.clamp(torch.tensor(8.0*pitch_angle + 3.0*pitch_rate), -1.0, 1.0))

        if step < 100:
            drive_cmd = 0.0
            phase = "STAND"
        elif step < 200:
            drive_cmd = 0.5
            phase = "FWD  "
        else:
            drive_cmd = -0.5
            phase = "BWD  "

        wheel_cmd = float(torch.clamp(torch.tensor(balance_cmd + drive_cmd), -1.0, 1.0))
        action = torch.zeros(1, action_dim, device="cuda:0")
        action[0, 2] = wheel_cmd
        action[0, 3] = wheel_cmd

        obs, total_reward, terminated, truncated, info = env.step(action)
        s = get_state(robot, env)

        # episode_sums accumulates each step; delta = per-step reward contribution
        curr_sums = {k: v[0].item() for k, v in reward_mgr.episode_sums.items()}
        if prev_sums is None or terminated[0] or truncated[0]:
            term_rewards = curr_sums
        else:
            term_rewards = {k: curr_sums[k] - prev_sums.get(k, 0.0) for k in curr_sums}
        prev_sums = None if (terminated[0] or truncated[0]) else curr_sums

        if step % 20 == 0:
            vel_w = robot.data.root_lin_vel_w[0]
            vx, vy = vel_w[0].item(), vel_w[1].item()
            print(f"\n  [{phase} step {step:3d}]  h={s['height']:.3f}m  tilt={s['tilt_deg']:.1f}°  "
                  f"vx={vx:+.3f}  vy={vy:+.3f}  total_R={total_reward[0].item():.4f}")
            for name, val in term_rewards.items():
                bar = "█" * min(int(abs(val) * 30), 25)
                sign = "+" if val >= 0 else ""
                print(f"    {name:35s} {sign}{val:.4f}  {bar}")

        if terminated[0] or truncated[0]:
            print(f"\n  *** TERMINATED at step {step} | tilt={s['tilt_deg']:.1f}° ***\n")

    return env


# ─────────────────────────────────────────────────────────────────────────────
def make_imu_marker():
    """Tạo VisualizationMarkers để vẽ frame axes (X=đỏ, Y=xanh lá, Z=xanh dương)."""
    cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/imu_frame")
    cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    return VisualizationMarkers(cfg)


def update_imu_marker(marker, env):
    """Vẽ frame IMU = robot_quat ⊗ offset_quat.

    imu.data KHÔNG chứa quat_w — chỉ có ang_vel_b / lin_acc_b.
    Offset phải compose thủ công với root quat của robot.
    """
    from isaaclab.utils.math import quat_mul

    robot = env.scene["robot"]
    robot_pos  = robot.data.root_pos_w   # (num_envs, 3)
    robot_quat = robot.data.root_quat_w  # (num_envs, 4) w,x,y,z

    # Offset: -90° around Z = (w=0.7071, x=0, y=0, z=-0.7071)
    # Lấy từ ImuCfg.OffsetCfg.rot được set trong legged_v3_wheel_env_cfg.py
    offset_quat = torch.tensor(
        [[0.7071, 0.0, 0.0, -0.7071]], device=robot_quat.device
    ).expand(robot_quat.shape[0], -1)

    # IMU frame trong world = robot body frame ⊗ offset
    imu_quat_w = quat_mul(robot_quat, offset_quat)

    marker.visualize(translations=robot_pos, orientations=imu_quat_w)


def keep_alive(env):
    """Keep simulation window open, hiển thị IMU axes liên tục."""
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros(1, action_dim, device="cuda:0")
    marker = make_imu_marker()
    print("\nIMU axes: RED=X  GREEN=Y  BLUE=Z (xanh dương)")
    print("Close the window to exit.")
    while simulation_app.is_running():
        env.step(zero_action)
        update_imu_marker(marker, env)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scenario = args.scenario
    last_env = None

    if scenario in (0, 4):
        last_env = run_contact_check()
    if scenario in (0, 1):
        if last_env is not None:
            last_env.close()
        last_env = run_zero_actions()
    if scenario in (0, 2):
        if last_env is not None:
            last_env.close()
        last_env = run_fixed_hip()
    if scenario in (0, 3):
        if last_env is not None:
            last_env.close()
        last_env = run_manual_balance(kp_fwd=25.0, kd_fwd=0.3, kp_roll=3.0, kd_roll=0.2)
        last_env.close()
        last_env = run_manual_balance(kp_fwd=35.0, kd_fwd=0.5, kp_roll=5.0, kd_roll=0.3)
    if scenario in (0, 5):
        if last_env is not None:
            last_env.close()
        last_env = run_direction_test()
    if scenario in (0, 6):
        if last_env is not None:
            last_env.close()
        last_env = run_reward_check()

    if last_env is not None:
        keep_alive(last_env)

    simulation_app.close()
