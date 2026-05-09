# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Rate PID Control — điều khiển tốc độ góc (angular velocity) cho quadcopter.

VÒNG TRONG CÙNG CỦA CASCADE — chạy nhanh nhất (400-1000Hz trong thực tế).

Mục đích: Tune Rate PID TRƯỚC HẾT — đây là vòng quan trọng nhất!
- Setpoint: tốc độ góc mong muốn p_des, q_des, r_des (rad/s)
- Feedback: từ gyroscope (angular velocity)
- Output trực tiếp: moment (roll, pitch, yaw)
- KHÔNG có attitude feedback (chỉ rate)

Vòng điều khiển:
    desired_roll_rate, desired_pitch_rate, desired_yaw_rate (setpoint)
         ↓
    Rate PID (rate error → moment)
         ↓
    Allocation matrix (moment → motor forces)
         ↓
    UAV

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/02_uav_pid_rate.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from collections import deque

import torch
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rate PID control (angular velocity) cho Crazyflie.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.uav.uav_cfg import UAV_CFG
from pid_controller import PIDController, make_alloc_inv

# === CẤU HÌNH THAM SỐ ===
class RateConfig:
    """Cấu hình điều khiển tốc độ góc (angular rate)"""
    
    # Target angular rates (rad/s)
    # Roll rate (p) - xoay quanh trục X
    TARGET_ROLL_RATE = math.radians(30.0)   # 30 độ/giây
    
    # Pitch rate (q) - xoay quanh trục Y
    TARGET_PITCH_RATE = math.radians(45.0)  # 45 độ/giây
    
    # Yaw rate (r) - xoay quanh trục Z
    TARGET_YAW_RATE = math.radians(60.0)    # 60 độ/giây
    
    # Hoặc dùng step test (thay đổi setpoint sau 5 giây)
    # STEP_CHANGE_TIME = 5.0  # seconds
    # STEP_ROLL_RATE = math.radians(50.0)
    
    # Rate PID gains (p, q, r)
    # Đây là vòng trong cùng — gains phải cao để tracking nhanh
    # Tham khảo: Crazyflie rate PID thường dùng kp=0.1-0.5, ki=0.05-0.2, kd=0.002-0.01
    RATE_ROLL_KP = 0.0025
    RATE_ROLL_KI = 0.01
    RATE_ROLL_KD = 0.0005
    RATE_ROLL_INTEGRAL_LIMIT = 0.5
    
    RATE_PITCH_KP = 0.025
    RATE_PITCH_KI = 0.01
    RATE_PITCH_KD = 0.0005
    RATE_PITCH_INTEGRAL_LIMIT = 0.5
    
    RATE_YAW_KP = 0.002
    RATE_YAW_KI = 0.008
    RATE_YAW_KD = 0.004
    RATE_YAW_INTEGRAL_LIMIT = 0.3
    
    # Thrust để giữ hover
    HOVER_THRUST = 0.35  # N
    
    # Giới hạn moment output
    MAX_ROLL_MOMENT = 0.03   # Nm
    MAX_PITCH_MOMENT = 0.03  # Nm
    MAX_YAW_MOMENT = 0.02    # Nm
    
    # Nhiễu
    ENABLE_NOISE = False
    GYRO_NOISE_STD = 0.01  # rad/s noise trên gyro

class DisplayConfig:
    """Cấu hình hiển thị"""
    WINDOW_SIZE = 10.0      # seconds
    PLOT_INTERVAL = 5
    CAMERA_OFFSET = [-1.5, -1.5, 0.8]

# === CLASS TÍNH TOÁN TỐC ĐỘ GÓC ===
def get_angular_velocity_from_quat(quat, prev_quat, dt):
    """
    Tính tốc độ góc (angular velocity) từ quaternion.
    Công thức: ω = 2 * q̇ * q⁻¹
    Hoặc dùng finite difference đơn giản hơn.
    """
    # Cách đơn giản: dùng Euler angles rate (chỉ đủ tốt cho góc nhỏ)
    # Trong thực tế, simulator có thể cung cấp angular velocity trực tiếp
    # Robot.data.root_ang_vel_w — nên dùng cái này!
    pass

# === CLASS ĐỒ THỊ RATE ===
class RatePlotter:
    """Đồ thị 3 hàng dọc, mỗi hàng chồng rate (actual/target) và error"""
    
    def __init__(self, window_size, dt):
        self.window_size = window_size
        self.max_history = int(window_size / dt) + 50
        
        # Buffers
        self.time_buffer = deque(maxlen=self.max_history)
        self.roll_rate_buffer = deque(maxlen=self.max_history)
        self.pitch_rate_buffer = deque(maxlen=self.max_history)
        self.yaw_rate_buffer = deque(maxlen=self.max_history)
        self.target_roll_buffer = deque(maxlen=self.max_history)
        self.target_pitch_buffer = deque(maxlen=self.max_history)
        self.target_yaw_buffer = deque(maxlen=self.max_history)
        self.roll_err_buffer = deque(maxlen=self.max_history)
        self.pitch_err_buffer = deque(maxlen=self.max_history)
        self.yaw_err_buffer = deque(maxlen=self.max_history)
        
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        self.actual_lines = []
        self.target_lines = []
        self.error_lines = []
        self._setup_plots()
        
    def _setup_plots(self):
        titles = ["Roll Rate (p)", "Pitch Rate (q)", "Yaw Rate (r)"]
        for i, ax in enumerate(self.axes):
            ax.set_title(titles[i], fontsize=11, fontweight='bold')
            ax.set_ylabel("Rate [deg/s]")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Vẽ 3 đường trên cùng 1 plot
            line_actual, = ax.plot([], [], 'b-', lw=2, label='Actual rate')
            line_target, = ax.plot([], [], 'r--', lw=1.5, label='Target rate')
            line_error, = ax.plot([], [], 'g:', lw=1.2, label='Error')
            
            self.actual_lines.append(line_actual)
            self.target_lines.append(line_target)
            self.error_lines.append(line_error)
            ax.legend(loc='upper right', fontsize=8)
        
        self.axes[-1].set_xlabel("Time [s]")
        self.fig.suptitle("Angular Rate Control (Rate PID)", fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        
    def update(self, current_time, 
               roll_rate_deg, pitch_rate_deg, yaw_rate_deg,
               target_roll_deg, target_pitch_deg, target_yaw_deg,
               roll_err_deg, pitch_err_deg, yaw_err_deg):
        
        self.time_buffer.append(current_time)
        self.roll_rate_buffer.append(roll_rate_deg)
        self.pitch_rate_buffer.append(pitch_rate_deg)
        self.yaw_rate_buffer.append(yaw_rate_deg)
        self.target_roll_buffer.append(target_roll_deg)
        self.target_pitch_buffer.append(target_pitch_deg)
        self.target_yaw_buffer.append(target_yaw_deg)
        self.roll_err_buffer.append(roll_err_deg)
        self.pitch_err_buffer.append(pitch_err_deg)
        self.yaw_err_buffer.append(yaw_err_deg)
        
        if not self.time_buffer:
            return
        
        t = list(self.time_buffer)
        
        # Cập nhật 3 subplot
        self.actual_lines[0].set_data(t, list(self.roll_rate_buffer))
        self.target_lines[0].set_data(t, list(self.target_roll_buffer))
        self.error_lines[0].set_data(t, list(self.roll_err_buffer))
        
        self.actual_lines[1].set_data(t, list(self.pitch_rate_buffer))
        self.target_lines[1].set_data(t, list(self.target_pitch_buffer))
        self.error_lines[1].set_data(t, list(self.pitch_err_buffer))
        
        self.actual_lines[2].set_data(t, list(self.yaw_rate_buffer))
        self.target_lines[2].set_data(t, list(self.target_yaw_buffer))
        self.error_lines[2].set_data(t, list(self.yaw_err_buffer))
        
        # Xác định giới hạn trục Y
        all_vals = (list(self.roll_rate_buffer) + list(self.pitch_rate_buffer) + list(self.yaw_rate_buffer) +
                    list(self.target_roll_buffer) + list(self.target_pitch_buffer) + list(self.target_yaw_buffer) +
                    list(self.roll_err_buffer) + list(self.pitch_err_buffer) + list(self.yaw_err_buffer))
        if all_vals:
            margin = 40.0
            y_min = min(all_vals) - margin
            y_max = max(all_vals) + margin
            for ax in self.axes:
                ax.set_xlim(max(0.0, t[-1] - self.window_size), t[-1] + 0.5)
                ax.set_ylim(y_min, y_max)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# === CLASS ĐIỀU KHIỂN RATE ===
class RateController:
    """Điều khiển tốc độ góc (angular rate) - Vòng trong cùng"""
    
    def __init__(self, dt):
        self.dt = dt
        
        # Roll rate (p) PID
        self.roll_rate_pid = PIDController(
            kp=RateConfig.RATE_ROLL_KP,
            ki=RateConfig.RATE_ROLL_KI,
            kd=RateConfig.RATE_ROLL_KD,
            integral_limit=RateConfig.RATE_ROLL_INTEGRAL_LIMIT
        )
        
        # Pitch rate (q) PID
        self.pitch_rate_pid = PIDController(
            kp=RateConfig.RATE_PITCH_KP,
            ki=RateConfig.RATE_PITCH_KI,
            kd=RateConfig.RATE_PITCH_KD,
            integral_limit=RateConfig.RATE_PITCH_INTEGRAL_LIMIT
        )
        
        # Yaw rate (r) PID
        self.yaw_rate_pid = PIDController(
            kp=RateConfig.RATE_YAW_KP,
            ki=RateConfig.RATE_YAW_KI,
            kd=RateConfig.RATE_YAW_KD,
            integral_limit=RateConfig.RATE_YAW_INTEGRAL_LIMIT
        )
        
    def reset(self):
        """Reset all PIDs"""
        self.roll_rate_pid.reset()
        self.pitch_rate_pid.reset()
        self.yaw_rate_pid.reset()
        
    def compute_moments(self, current_roll_rate, current_pitch_rate, current_yaw_rate,
                        target_roll_rate, target_pitch_rate, target_yaw_rate):
        """Tính moment từ sai số tốc độ góc"""
        
        # Roll rate error (rad/s)
        roll_rate_error = target_roll_rate - current_roll_rate
        roll_moment = self.roll_rate_pid.update(roll_rate_error, self.dt)
        
        # Pitch rate error
        pitch_rate_error = target_pitch_rate - current_pitch_rate
        pitch_moment = self.pitch_rate_pid.update(pitch_rate_error, self.dt)
        
        # Yaw rate error
        yaw_rate_error = target_yaw_rate - current_yaw_rate
        yaw_moment = self.yaw_rate_pid.update(yaw_rate_error, self.dt)
        
        # Limit moments
        roll_moment = max(-RateConfig.MAX_ROLL_MOMENT, 
                         min(RateConfig.MAX_ROLL_MOMENT, roll_moment))
        pitch_moment = max(-RateConfig.MAX_PITCH_MOMENT, 
                          min(RateConfig.MAX_PITCH_MOMENT, pitch_moment))
        yaw_moment = max(-RateConfig.MAX_YAW_MOMENT, 
                        min(RateConfig.MAX_YAW_MOMENT, yaw_moment))
        
        return roll_moment, pitch_moment, yaw_moment, {
            'roll_error': roll_rate_error,
            'pitch_error': pitch_rate_error,
            'yaw_error': yaw_rate_error
        }

# === HÀM KHỞI TẠO ===
def setup_simulation():
    """Khởi tạo môi trường simulation"""
    sim_config = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    simulation = SimulationContext(sim_config)
    simulation.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])
    
    # Ground plane
    ground_config = sim_utils.GroundPlaneCfg()
    ground_config.func("/World/defaultGroundPlane", ground_config)
    
    # Lighting
    light_config = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_config.func("/World/Light", light_config)
    
    # Robot
    robot_config = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_config = robot_config.replace(
        init_state=robot_config.init_state.replace(pos=(0.0, 0.0, 0.3))
    )
    robot_config.spawn.func("/World/Crazyflie", robot_config.spawn, 
                           translation=robot_config.init_state.pos)
    robot = Articulation(robot_config)
    
    simulation.reset()
    
    return simulation, robot

# === HÀM MAIN ===
def main():
    print("=" * 70)
    print("RATE PID CONTROL - Angular Velocity Control (Vòng trong cùng của Cascade)")
    print("=" * 70)
    
    # Setup simulation
    simulation, robot = setup_simulation()
    
    # Get propeller IDs and allocation matrix
    propeller_body_ids = robot.find_bodies("m.*_prop")[0]
    allocation_matrix_inv = make_alloc_inv(device=simulation.device)
    
    # Setup controller
    sim_dt = simulation.get_physics_dt()
    controller = RateController(sim_dt)
    plotter = RatePlotter(DisplayConfig.WINDOW_SIZE, sim_dt)
    
    # Target angular rates (rad/s -> deg/s for display)
    target_roll_rate_rad = RateConfig.TARGET_ROLL_RATE
    target_pitch_rate_rad = RateConfig.TARGET_PITCH_RATE
    target_yaw_rate_rad = RateConfig.TARGET_YAW_RATE
    
    target_roll_rate_deg = math.degrees(target_roll_rate_rad)
    target_pitch_rate_deg = math.degrees(target_pitch_rate_rad)
    target_yaw_rate_deg = math.degrees(target_yaw_rate_rad)
    
    print(f"\n[INFO] Target Angular Rates:")
    print(f"       Roll Rate (p):  {target_roll_rate_deg:+.1f}°/s ({target_roll_rate_rad:+.3f} rad/s)")
    print(f"       Pitch Rate (q): {target_pitch_rate_deg:+.1f}°/s ({target_pitch_rate_rad:+.3f} rad/s)")
    print(f"       Yaw Rate (r):   {target_yaw_rate_deg:+.1f}°/s ({target_yaw_rate_rad:+.3f} rad/s)")
    print(f"\n[INFO] Rate PID Gains:")
    print(f"       Roll:  KP={RateConfig.RATE_ROLL_KP}, KI={RateConfig.RATE_ROLL_KI}, KD={RateConfig.RATE_ROLL_KD}")
    print(f"       Pitch: KP={RateConfig.RATE_PITCH_KP}, KI={RateConfig.RATE_PITCH_KI}, KD={RateConfig.RATE_PITCH_KD}")
    print(f"       Yaw:   KP={RateConfig.RATE_YAW_KP}, KI={RateConfig.RATE_YAW_KI}, KD={RateConfig.RATE_YAW_KD}")
    print(f"\n[INFO] Hover thrust: {RateConfig.HOVER_THRUST:.3f} N")
    print(f"[INFO] Noise: {'ON' if RateConfig.ENABLE_NOISE else 'OFF'}")
    print("\n[INFO] Starting simulation... (Vòng RATE là vòng trong cùng, chạy nhanh nhất!)\n")
    
    # Simulation variables
    sim_time = 0.0
    step_count = 0
    log_interval = int(1.0 / sim_dt)
    
    # Main loop
    while simulation_app.is_running():
        # Get current state
        current_position = robot.data.root_pos_w[0]
        
        # Get angular velocity (rad/s) from robot data
        # robot.data.root_ang_vel_w gives [roll_rate, pitch_rate, yaw_rate] in rad/s
        ang_vel = robot.data.root_ang_vel_w[0]
        current_roll_rate_rad = ang_vel[0].item()
        current_pitch_rate_rad = ang_vel[1].item()
        current_yaw_rate_rad = ang_vel[2].item()
        
        # === Rate Control: rate error → moment ===
        roll_moment, pitch_moment, yaw_moment, errors = controller.compute_moments(
            current_roll_rate_rad, current_pitch_rate_rad, current_yaw_rate_rad,
            target_roll_rate_rad, target_pitch_rate_rad, target_yaw_rate_rad
        )
        
        # Constant thrust để giữ hover
        thrust = RateConfig.HOVER_THRUST
        
        # Add noise to gyro (simulate sensor noise)
        if RateConfig.ENABLE_NOISE:
            # Add noise to moments for more realistic response
            pass
        
        # === Motor Allocation ===
        wrench = torch.tensor([thrust, roll_moment, pitch_moment, yaw_moment], 
                             device=simulation.device)
        motor_forces = (allocation_matrix_inv @ wrench).clamp(min=0.0)
        
        # Apply forces to motors
        forces_tensor = torch.zeros(robot.num_instances, 4, 3, device=simulation.device)
        forces_tensor[0, :, 2] = motor_forces
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_tensor,
            torques=torch.zeros_like(forces_tensor),
            body_ids=propeller_body_ids,
        )
        
        # Step simulation
        robot.write_data_to_sim()
        simulation.step()
        sim_time += sim_dt
        step_count += 1
        robot.update(sim_dt)
        
        # Convert to degrees for display
        current_roll_rate_deg = math.degrees(current_roll_rate_rad)
        current_pitch_rate_deg = math.degrees(current_pitch_rate_rad)
        current_yaw_rate_deg = math.degrees(current_yaw_rate_rad)
        
        roll_err_deg = math.degrees(errors['roll_error'])
        pitch_err_deg = math.degrees(errors['pitch_error'])
        yaw_err_deg = math.degrees(errors['yaw_error'])
        
        # Logging
        if step_count % log_interval == 0:
            print(f"t={sim_time:5.1f}s | "
                  f"p={current_roll_rate_deg:+6.1f}°/s (err={roll_err_deg:+5.1f}) | "
                  f"q={current_pitch_rate_deg:+6.1f}°/s (err={pitch_err_deg:+5.1f}) | "
                  f"r={current_yaw_rate_deg:+6.1f}°/s (err={yaw_err_deg:+5.1f})")
        
        # Update plot
        if step_count % DisplayConfig.PLOT_INTERVAL == 0:
            plotter.update(
                sim_time,
                current_roll_rate_deg, current_pitch_rate_deg, current_yaw_rate_deg,
                target_roll_rate_deg, target_pitch_rate_deg, target_yaw_rate_deg,
                roll_err_deg, pitch_err_deg, yaw_err_deg
            )
        
        # Update camera
        uav_pos = current_position.cpu().numpy()
        simulation.set_camera_view(
            eye=[uav_pos[0] + DisplayConfig.CAMERA_OFFSET[0],
                 uav_pos[1] + DisplayConfig.CAMERA_OFFSET[1],
                 uav_pos[2] + DisplayConfig.CAMERA_OFFSET[2]],
            target=[uav_pos[0], uav_pos[1], uav_pos[2]],
        )
    
    print("\n[INFO] Simulation finished.")
    print("[INFO] Rate PID tuning done! Tiếp theo: Attitude Control (03_uav_pid_attitude.py)")

if __name__ == "__main__":
    main()
    simulation_app.close()