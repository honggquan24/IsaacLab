"""
Test script cho CRAZYFLIE_CFG từ isaaclab_assets/robots/quadcopter.py

Chạy:
    ./isaaclab.sh -p scripts/demos/test_crazyflie_cfg.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test CRAZYFLIE_CFG")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 1.5], target=[0.0, 0.0, 0.5])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0)
    )

    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    sim.reset()

    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    print("\n========== CRAZYFLIE_CFG Info ==========")
    print(f"  Num joints  : {robot.num_joints}")
    print(f"  Joint names : {robot.joint_names}")
    print(f"  Joint vel   : {robot.data.default_joint_vel[0].tolist()}")
    print(f"  Body names  : {robot.body_names}")
    print(f"  Mass        : {robot_mass.item():.4f} kg")
    print("========================================\n")

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    prop_angles = robot.data.default_joint_pos.clone()
    prop_vel = robot.data.default_joint_vel.clone()

    while simulation_app.is_running():
        if count % 20000 == 0 and count > 0:
            sim_time = 0.0
            count = 0
            prop_angles = robot.data.default_joint_pos.clone()
            robot.write_joint_state_to_sim(prop_angles, prop_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            print(">>>>>>>> Reset!")

        thrust_scale = min(1.0, sim_time / 5.0)
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = thrust_scale * robot_mass * gravity / 4.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=prop_body_ids
        )

        prop_angles += prop_vel * sim_dt
        robot.write_joint_state_to_sim(prop_angles, prop_vel)
        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        robot.update(sim_dt)

        if count % int(1.0 / sim_dt) == 0:
            pos = robot.data.root_pos_w[0]
            print(f"t={sim_time:5.1f}s | pos=({pos[0].item():+.3f}, {pos[1].item():+.3f}, {pos[2].item():+.3f}) | thrust_scale={thrust_scale:.2f}")

        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 0.8, p[1] - 0.8, p[2] + 0.8],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
