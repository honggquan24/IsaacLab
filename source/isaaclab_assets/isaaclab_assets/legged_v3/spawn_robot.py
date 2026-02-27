"""Minimal script to spawn Legged V3 robot and run simulation loop.

No RL env, no rewards, no actions — just spawn and watch.

Run with:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/legged_v3/spawn_robot.py

Close the window to exit.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Isaac Sim booted — now import other modules
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.legged_v3.legged_v3_cfg import LEGGED_ROBOT_V3_CFG

# ── Simulation setup ──────────────────────────────────────────────────────────
sim_cfg = sim_utils.SimulationCfg(dt=1 / 60.0, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)

sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])

# ── Scene ─────────────────────────────────────────────────────────────────────
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

cfg = LEGGED_ROBOT_V3_CFG.replace(prim_path="/World/Robot")
robot = Articulation(cfg)

# ── Reset ─────────────────────────────────────────────────────────────────────
sim.reset()
robot.reset()

print("\n" + "=" * 50)
print("Robot spawned. Running sim loop — close window to exit.")
print(f"Joints ({robot.num_joints}): {robot.data.joint_names}")
print(f"Bodies ({robot.num_bodies}): {robot.data.body_names}")
print("=" * 50 + "\n")

# ── Sim loop ──────────────────────────────────────────────────────────────────
while simulation_app.is_running():
    # zero effort (robot just sits/falls under gravity)
    zero_effort = torch.zeros(1, robot.num_joints, device=sim_cfg.device)
    robot.set_joint_effort_target(zero_effort)
    robot.write_data_to_sim()

    sim.step()
    robot.update(sim.get_physics_dt())

simulation_app.close()
