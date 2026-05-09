# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Attitude PID Control — điều khiển góc quay (roll, pitch, yaw) cho quadcopter.

Mục đích: Tune attitude PID TRƯỚC KHI thêm position control.
- Giữ drone hover tại một điểm (position hold)
- Điều khiển trực tiếp các góc desired_roll, desired_pitch, desired_yaw
- Chỉ có attitude PID (không có position feedback x,y)

Vòng điều khiển:
    desired_roll/desired_pitch/desired_yaw (setpoint)
         ↓
    Attitude PID (angle error → moment)
         ↓
    Allocation matrix (moment → motor forces)
         ↓
    UAV

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/02_uav_pid_attitude.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from collections import deque

import torch
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Attitude PID control cho Crazyflie - Roll/Pitch/Yaw only.")
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
class AttitudeConfig:
    """Cấu hình điều khiển attitude"""
    
    # Target angles (radians)
    TARGET_ROLL = math.radians(10.0)    # 10 độ
    TARGET_PITCH = math.radians(15.0)   # 15 độ  
    TARGET_YAW = math.radians(20.0)     # 20 độ
    
    # Hoặc có thể dùng oscillation test:
    # TARGET_ROLL = math.radians(5.0) * math.sin(time * 0.5)  # dao động sin
    
    # Attitude PID gains (roll, pitch, yaw)
    # Tham khảo: Crazyflie thường dùng kp=3-6, ki=0-0.5, kd=0.1-0.5
    ROLL_KP = 4.0
    ROLL_KI = 0.1
    ROLL_KD = 0.2
    ROLL_INTEGRAL_LIMIT = 0.5
    
    PITCH_KP = 4.0
    PITCH_KI = 0.1
    PITCH_KD = 0.2
    PITCH_INTEGRAL_LIMIT = 0.5
    
    YAW_KP = 3.0
    YAW_KI = 0.05
    YAW_KD = 0.15
    YAW_INTEGRAL_LIMIT = 0.3
    
    # Thrust để giữ hover (must be > mass * g)
    # Crazyflie mass = 0.033kg, g=9.81 => min_thrust = 0.324N
    HOVER_THRUST = 0.35  # N (hơi cao hơn trọng lực một chút)
    
    # Giới hạn moment
    MAX_ROLL_MOMENT = 0.03   # Nm
    MAX_PITCH_MOMENT = 0.03  # Nm
    MAX_YAW_MOMENT = 0.02    # Nm
    
    # Nhiễu motor (để test robustness)
    MOTOR_THRUST_NOISE = 1e-4   # N
    MOTOR_MOMENT_NOISE = 1e-5   # Nm
    ENABLE_NOISE = False         # Bật/tắt nhiễu

class DisplayConfig:
    """Cấu hình hiển thị"""
    WINDOW_SIZE = 15.0      # seconds
    PLOT_INTERVAL = 5       # steps
    CAMERA_OFFSET = [-1.5, -1.5, 0.8]

# === CLASS ĐỒ THỊ ATTITUDE ===
class AttitudePlotter:
    """Đồ thị cho roll, pitch, yaw"""
    
    def __init__(self, window_size, dt):
        self.window_size = window_size
        self.max_history = int(window_size / dt) + 50
        
        # Buffers cho thời gian và góc
        self.time_buffer = deque(maxlen=self.max_history)
        self.roll_buffer = deque(maxlen=self.max_history)
        self.pitch_buffer = deque(maxlen=self.max_history)
        self.yaw_buffer = deque(maxlen=self.max_history)
        
        self.target_roll_buffer = deque(maxlen=self.max_history)
        self.target_pitch_buffer = deque(maxlen=self.max_history)
        self.target_yaw_buffer = deque(maxlen=self.max_history)
        
        # Setup plot
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self._setup_plot()
        
    def _setup_plot(self):
        titles_angles = [
            ("Roll Angle", "deg"), 
            ("Pitch Angle", "deg"), 
            ("Yaw Angle", "deg")
        ]
        
        self.lines_actual = []
        self.lines_target = []
        
        for i, (ax, (title, unit)) in enumerate(zip(self.axes, titles_angles)):
            ax.set_title(f"{title} Response")
            ax.set_ylabel(f"{title} [{unit}]")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Đường actual và target
            line_actual, = ax.plot([], [], 'b-', linewidth=2, label='Actual')
            line_target, = ax.plot([], [], 'r--', linewidth=1.5, label='Target')
            self.lines_actual.append(line_actual)
            self.lines_target.append(line_target)
            
            ax.legend(loc='upper right', fontsize=8)
        
        self.axes[-1].set_xlabel("Time [s]")
        self.fig.tight_layout()
        
    def update(self, current_time, roll_deg, pitch_deg, yaw_deg, 
               target_roll_deg, target_pitch_deg, target_yaw_deg):
        """Cập nhật đồ thị"""
        
        # Add to buffers
        self.time_buffer.append(current_time)
        self.roll_buffer.append(roll_deg)
        self.pitch_buffer.append(pitch_deg)
        self.yaw_buffer.append(yaw_deg)
        self.target_roll_buffer.append(target_roll_deg)
        self.target_pitch_buffer.append(target_pitch_deg)
        self.target_yaw_buffer.append(target_yaw_deg)
        
        if not self.time_buffer:
            return
        
        t = list(self.time_buffer)
        
        # Update từng subplot
        # Roll
        self.lines_actual[0].set_data(t, list(self.roll_buffer))
        self.lines_target[0].set_data(t, list(self.target_roll_buffer))
        
        # Pitch
        self.lines_actual[1].set_data(t, list(self.pitch_buffer))
        self.lines_target[1].set_data(t, list(self.target_pitch_buffer))
        
        # Yaw
        self.lines_actual[2].set_data(t, list(self.yaw_buffer))
        self.lines_target[2].set_data(t, list(self.target_yaw_buffer))
        
        # Adjust axes limits
        all_angles = list(self.roll_buffer) + list(self.pitch_buffer) + list(self.yaw_buffer)
        all_targets = list(self.target_roll_buffer) + list(self.target_pitch_buffer) + list(self.target_yaw_buffer)
        all_values = all_angles + all_targets
        
        if all_values:
            margin = 15.0  # độ
            y_min = min(all_values) - margin
            y_max = max(all_values) + margin
            
            for ax in self.axes:
                ax.set_xlim(max(0.0, t[-1] - self.window_size), t[-1] + 0.5)
                ax.set_ylim(y_min, y_max)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# === CLASS ĐIỀU KHIỂN ATTITUDE ===
class AttitudeController:
    """Điều khiển attitude (roll, pitch, yaw) với PID"""
    
    def __init__(self, dt):
        self.dt = dt
        
        # Roll PID
        self.roll_pid = PIDController(
            kp=AttitudeConfig.ROLL_KP,
            ki=AttitudeConfig.ROLL_KI,
            kd=AttitudeConfig.ROLL_KD,
            integral_limit=AttitudeConfig.ROLL_INTEGRAL_LIMIT
        )
        
        # Pitch PID
        self.pitch_pid = PIDController(
            kp=AttitudeConfig.PITCH_KP,
            ki=AttitudeConfig.PITCH_KI,
            kd=AttitudeConfig.PITCH_KD,
            integral_limit=AttitudeConfig.PITCH_INTEGRAL_LIMIT
        )
        
        # Yaw PID (với wrap angle)
        self.yaw_pid = PIDController(
            kp=AttitudeConfig.YAW_KP,
            ki=AttitudeConfig.YAW_KI,
            kd=AttitudeConfig.YAW_KD,
            integral_limit=AttitudeConfig.YAW_INTEGRAL_LIMIT
        )
        
        # Lưu lỗi cũ cho derivative (nếu cần)
        self.prev_roll_error = 0.0
        self.prev_pitch_error = 0.0
        self.prev_yaw_error = 0.0
        
    def reset(self):
        """Reset all PID controllers"""
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.prev_roll_error = 0.0
        self.prev_pitch_error = 0.0
        self.prev_yaw_error = 0.0
        
    def angle_error_wrap(self, error):
        """Wrap angle error to [-pi, pi]"""
        return (error + math.pi) % (2 * math.pi) - math.pi
        
    def compute_moments(self, current_roll, current_pitch, current_yaw,
                        target_roll, target_pitch, target_yaw):
        """Tính moment từ sai số góc"""
        
        # Roll error
        roll_error = target_roll - current_roll
        roll_moment = self.roll_pid.update(roll_error, self.dt)
        
        # Pitch error
        pitch_error = target_pitch - current_pitch
        pitch_moment = self.pitch_pid.update(pitch_error, self.dt)
        
        # Yaw error (wrap to [-pi, pi])
        yaw_error = self.angle_error_wrap(target_yaw - current_yaw)
        yaw_moment = self.yaw_pid.update(yaw_error, self.dt)
        
        # Limit moments
        roll_moment = max(-AttitudeConfig.MAX_ROLL_MOMENT, 
                         min(AttitudeConfig.MAX_ROLL_MOMENT, roll_moment))
        pitch_moment = max(-AttitudeConfig.MAX_PITCH_MOMENT, 
                          min(AttitudeConfig.MAX_PITCH_MOMENT, pitch_moment))
        yaw_moment = max(-AttitudeConfig.MAX_YAW_MOMENT, 
                        min(AttitudeConfig.MAX_YAW_MOMENT, yaw_moment))
        
        return roll_moment, pitch_moment, yaw_moment

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
    print("=" * 60)
    print("ATTITUDE PID CONTROL - Crazyflie")
    print("=" * 60)
    
    # Setup simulation
    simulation, robot = setup_simulation()
    
    # Get propeller IDs and allocation matrix
    propeller_body_ids = robot.find_bodies("m.*_prop")[0]
    allocation_matrix_inv = make_alloc_inv(device=simulation.device)
    
    # Setup controller
    sim_dt = simulation.get_physics_dt()
    controller = AttitudeController(sim_dt)
    plotter = AttitudePlotter(DisplayConfig.WINDOW_SIZE, sim_dt)
    
    # Target angles in radians -> degrees for display
    target_roll_rad = AttitudeConfig.TARGET_ROLL
    target_pitch_rad = AttitudeConfig.TARGET_PITCH
    target_yaw_rad = AttitudeConfig.TARGET_YAW
    
    target_roll_deg = math.degrees(target_roll_rad)
    target_pitch_deg = math.degrees(target_pitch_rad)
    target_yaw_deg = math.degrees(target_yaw_rad)
    
    print(f"\n[INFO] Target angles:")
    print(f"       Roll:  {target_roll_deg:+.1f}° ({target_roll_rad:+.3f} rad)")
    print(f"       Pitch: {target_pitch_deg:+.1f}° ({target_pitch_rad:+.3f} rad)")
    print(f"       Yaw:   {target_yaw_deg:+.1f}° ({target_yaw_rad:+.3f} rad)")
    print(f"\n[INFO] Hover thrust: {AttitudeConfig.HOVER_THRUST:.3f} N")
    print(f"[INFO] Max moments: roll/pitch={AttitudeConfig.MAX_ROLL_MOMENT:.3f} Nm, yaw={AttitudeConfig.MAX_YAW_MOMENT:.3f} Nm")
    print(f"[INFO] Noise: {'ON' if AttitudeConfig.ENABLE_NOISE else 'OFF'}")
    print("\n[INFO] Starting simulation...\n")
    
    # Simulation variables
    sim_time = 0.0
    step_count = 0
    log_interval = int(1.0 / sim_dt)  # Log every second
    
    # Main loop
    while simulation_app.is_running():
        # Get current state
        current_position = robot.data.root_pos_w[0]
        quaternion = robot.data.root_quat_w[0]
        
        # Get Euler angles (radians)
        roll, pitch, yaw = [angle[0].item() for angle in 
                           euler_xyz_from_quat(quaternion.unsqueeze(0))]
        
        # === Attitude Control ===
        roll_moment, pitch_moment, yaw_moment = controller.compute_moments(
            roll, pitch, yaw,
            target_roll_rad, target_pitch_rad, target_yaw_rad
        )
        
        # Constant thrust để giữ hover
        thrust = AttitudeConfig.HOVER_THRUST
        
        # === Motor Allocation ===
        wrench = torch.tensor([thrust, roll_moment, pitch_moment, yaw_moment], 
                             device=simulation.device)
        motor_forces = (allocation_matrix_inv @ wrench).clamp(min=0.0)
        
        # Add noise (optional)
        if AttitudeConfig.ENABLE_NOISE:
            motor_forces += torch.randn(4, device=simulation.device) * AttitudeConfig.MOTOR_THRUST_NOISE
        
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
        
        # Logging
        if step_count % log_interval == 0:
            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            
            roll_error_deg = math.degrees(target_roll_rad - roll)
            pitch_error_deg = math.degrees(target_pitch_rad - pitch)
            yaw_error_deg = math.degrees(controller.angle_error_wrap(target_yaw_rad - yaw))
            
            print(f"t={sim_time:5.1f}s | "
                  f"roll={roll_deg:+6.1f}° (err={roll_error_deg:+5.1f}°) | "
                  f"pitch={pitch_deg:+6.1f}° (err={pitch_error_deg:+5.1f}°) | "
                  f"yaw={yaw_deg:+6.1f}° (err={yaw_error_deg:+5.1f}°)")
        
        # Update plot (convert to degrees)
        if step_count % DisplayConfig.PLOT_INTERVAL == 0:
            plotter.update(
                sim_time,
                math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                target_roll_deg, target_pitch_deg, target_yaw_deg
            )
        
        # Update camera to follow drone
        uav_pos = current_position.cpu().numpy()
        simulation.set_camera_view(
            eye=[uav_pos[0] + DisplayConfig.CAMERA_OFFSET[0],
                 uav_pos[1] + DisplayConfig.CAMERA_OFFSET[1],
                 uav_pos[2] + DisplayConfig.CAMERA_OFFSET[2]],
            target=[uav_pos[0], uav_pos[1], uav_pos[2]],
        )
    
    print("\n[INFO] Simulation finished.")
    print("[INFO] Close plot window to exit.")

if __name__ == "__main__":
    main()
    simulation_app.close()