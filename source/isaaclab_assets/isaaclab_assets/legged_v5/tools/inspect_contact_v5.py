"""Chẩn đoán: kiểm tra RigidBodyAPI và PhysxContactReportAPI trên từng prim.

Run:
    ./isaaclab.sh -p .../diag_contact_v5.py --headless
"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets.legged_v5.legged_v5_cfg import LEGGED_V5_CFG

sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200.0, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

robot_cfg = LEGGED_V5_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)
sim.reset()

import omni.usd
from pxr import UsdPhysics, PhysxSchema, Usd

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/Robot")

print("\n" + "="*72)
print("PRIM SCAN — RigidBodyAPI (R) | PhysxContactReportAPI (C)")
print("="*72)

for prim in Usd.PrimRange(root):
    path = str(prim.GetPath())
    has_rb  = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    has_cr  = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
    has_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    flags = ("R" if has_rb else "-") + ("C" if has_cr else "-") + ("A" if has_art else "-")
    if has_rb or has_cr or has_art:
        print(f"  [{flags}] {path}")

print("="*72)
print("R=RigidBodyAPI  C=PhysxContactReportAPI  A=ArticulationRootAPI")
print("="*72 + "\n")

simulation_app.close()
