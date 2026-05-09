# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Altitude PID Control — điều khiển độ cao Z bằng PID.
Đầu ra PID điều khiển trực tiếp lực/moment vật lý (N, Nm).

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/01_uav_pid_altitude.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from collections import deque

import matplotlib.pyplot as plt
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Altitude PID control — direct force.")
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

# === CẤU HÌNH THAM SỐ ĐIỀU KHIỂN ===
class ControlConfig:
    """Cấu hình tham số điều khiển cho UAV"""
    # Mục tiêu
    TARGET_ALTITUDE = 1.0  # độ cao mục tiêu [m]
    
    # PID cho độ cao
    ALT_KP = 1.5
    ALT_KI = 0.7
    ALT_KD = 0.12
    ALT_INTEGRAL_LIMIT = 1.0
    
    # PID cho attitude (roll, pitch, yaw)
    ATT_KP = 0.001
    ATT_KI = 0.001
    ATT_KD = 0.002
    ATT_INTEGRAL_LIMIT = 0.01
    
    # PID cho yaw
    YAW_KP = 0.001
    YAW_KI = 0.0
    YAW_KD = 0.001
    
    # Nhiễu motor
    MOTOR_THRUST_NOISE = 1e-5  # [N]
    MOTOR_MOMENT_NOISE = 1e-5  # [Nm]

# === CẤU HÌNH HIỂN THỊ ===
class DisplayConfig:
    """Cấu hình hiển thị và đồ thị"""
    WINDOW_SIZE = 20.0  # giây
    PLOT_INTERVAL = 5   # số bước cập nhật
    CAMERA_OFFSET = [-1.0, -1.0, 1.0]  # [x, y, z] so với UAV

# === LỚP QUẢN LÝ ĐỒ THỊ ===
class AltitudePlotter:
    """Quản lý đồ thị độ cao thời gian thực"""
    
    def __init__(self, window_size, sim_dt):
        self.window_size = window_size
        self.max_history = int(window_size / sim_dt) + 50
        
        # Khởi tạo bộ đệm dữ liệu
        self.time_buffer = deque(maxlen=self.max_history)
        self.altitude_buffer = deque(maxlen=self.max_history)
        self.target_altitude_buffer = deque(maxlen=self.max_history)
        self.error_buffer = deque(maxlen=self.max_history)
        
        # Tạo figure và axes
        plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(9, 4))
        self._setup_plot()
        
    def _setup_plot(self):
        """Cấu hình giao diện đồ thị"""
        self.axes.set_title("UAV Altitude Response — Z(t)")
        self.axes.set_xlabel("Time [s]")
        self.axes.set_ylabel("Height [m]")
        self.axes.grid(True)
        
        # Các đường đồ thị
        self.altitude_line, = self.axes.plot([], [], "b-", lw=2, label="Actual altitude")
        self.target_line, = self.axes.plot([], [], "r--", lw=1.5, label="Target altitude")
        self.error_line, = self.axes.plot([], [], "g:", lw=1, label="Error")
        
        self.axes.legend(loc="upper right")
        self.figure.tight_layout()
    
    def update(self, current_time, altitude, target_altitude):
        """Cập nhật đồ thị với dữ liệu mới"""
        # Tính toán error
        error = target_altitude - altitude
        
        # Thêm vào buffer
        self.time_buffer.append(current_time)
        self.altitude_buffer.append(altitude)
        self.target_altitude_buffer.append(target_altitude)
        self.error_buffer.append(error)
        
        # Cập nhật đồ thị
        self.altitude_line.set_data(list(self.time_buffer), list(self.altitude_buffer))
        self.target_line.set_data(list(self.time_buffer), list(self.target_altitude_buffer))
        self.error_line.set_data(list(self.time_buffer), list(self.error_buffer))
        
        # Điều chỉnh giới hạn trục
        if self.time_buffer:
            self.axes.set_xlim(max(0.0, self.time_buffer[-1] - self.window_size), 
                              self.time_buffer[-1] + 0.5)
            
            all_values = list(self.altitude_buffer) + list(self.target_altitude_buffer) + list(self.error_buffer)
            margin = 0.3
            self.axes.set_ylim(min(all_values) - margin, max(all_values) + margin)
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

# === LỚP ĐIỀU KHIỂN UAV ===
class UAVAltitudeController:
    """Điều khiển độ cao cho UAV sử dụng PID"""
    
    def __init__(self, simulation, robot, prop_body_ids, allocation_matrix_inv):
        self.simulation = simulation
        self.robot = robot
        self.prop_body_ids = prop_body_ids
        self.allocation_matrix_inv = allocation_matrix_inv
        self.sim_dt = simulation.get_physics_dt()
        
        # Khởi tạo bộ điều khiển PID
        self._init_pid_controllers()
        
    def _init_pid_controllers(self):
        """Khởi tạo các bộ PID cho độ cao và attitude"""
        # Điều khiển độ cao
        self.altitude_pid = PIDController(
            kp=ControlConfig.ALT_KP,
            ki=ControlConfig.ALT_KI,
            kd=ControlConfig.ALT_KD,
            integral_limit=ControlConfig.ALT_INTEGRAL_LIMIT
        )
        
        # Điều khiển attitude
        self.roll_pid = PIDController(
            kp=ControlConfig.ATT_KP,
            ki=ControlConfig.ATT_KI,
            kd=ControlConfig.ATT_KD,
            integral_limit=ControlConfig.ATT_INTEGRAL_LIMIT
        )
        
        self.pitch_pid = PIDController(
            kp=ControlConfig.ATT_KP,
            ki=ControlConfig.ATT_KI,
            kd=ControlConfig.ATT_KD,
            integral_limit=ControlConfig.ATT_INTEGRAL_LIMIT
        )
        
        self.yaw_pid = PIDController(
            kp=ControlConfig.YAW_KP,
            ki=ControlConfig.YAW_KI,
            kd=ControlConfig.YAW_KD,
            integral_limit=ControlConfig.ATT_INTEGRAL_LIMIT
        )
    
    def get_current_state(self):
        """Lấy trạng thái hiện tại của UAV"""
        position = self.robot.data.root_pos_w[0]
        velocity = self.robot.data.root_lin_vel_w[0]
        quaternion = self.robot.data.root_quat_w[0]
        
        # Tính Euler angles
        roll, pitch, yaw = [angle[0].item() for angle in euler_xyz_from_quat(quaternion.unsqueeze(0))]
        
        return {
            'position': position,
            'velocity': velocity,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
    
    def compute_thrust_from_altitude(self, current_altitude, target_altitude):
        """Tính lực đẩy từ sai số độ cao"""
        altitude_error = target_altitude - current_altitude
        thrust = self.altitude_pid.update(altitude_error, self.sim_dt)
        return max(0.0, thrust), altitude_error
    
    def compute_moments_from_attitude(self, current_roll, current_pitch, current_yaw):
        """Tính moment từ sai số attitude (giữ UAV thăng bằng)"""
        # Yaw error với wrap-around
        yaw_error = (0.0 - current_yaw + math.pi) % (2 * math.pi) - math.pi
        
        roll_moment = self.roll_pid.update(0.0 - current_roll, self.sim_dt)
        pitch_moment = self.pitch_pid.update(0.0 - current_pitch, self.sim_dt)
        yaw_moment = self.yaw_pid.update(yaw_error, self.sim_dt)
        
        return roll_moment, pitch_moment, yaw_moment
    
    def apply_forces_to_motors(self, thrust, roll_moment, pitch_moment, yaw_moment):
        """Phân bổ lực/moment cho các motor riêng lẻ"""
        # Tạo vector wrench: [lực đẩy, moment roll, moment pitch, moment yaw]
        wrench = torch.tensor([thrust, roll_moment, pitch_moment, yaw_moment], 
                             device=self.simulation.device)
        
        # Tính lực cho từng motor
        motor_forces = (self.allocation_matrix_inv @ wrench).clamp(min=0.0)
        
        # Có thể thêm nhiễu nếu cần
        # motor_forces += torch.randn(4, device=self.simulation.device) * ControlConfig.MOTOR_THRUST_NOISE
        
        # Gán lực cho từng motor (chỉ theo trục Z)
        forces_tensor = torch.zeros(self.robot.num_instances, 4, 3, device=self.simulation.device)
        forces_tensor[0, :, 2] = motor_forces
        
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_tensor,
            torques=torch.zeros_like(forces_tensor),
            body_ids=self.prop_body_ids,
        )
    
    def control_step(self, target_altitude):
        """Thực hiện một bước điều khiển"""
        # Lấy trạng thái hiện tại
        state = self.get_current_state()
        current_altitude = state['position'][2].item()
        
        # Tính lực đẩy từ độ cao
        thrust, altitude_error = self.compute_thrust_from_altitude(current_altitude, target_altitude)
        
        # Tính moment từ attitude
        roll_moment, pitch_moment, yaw_moment = self.compute_moments_from_attitude(
            state['roll'], state['pitch'], state['yaw']
        )
        
        # Áp dụng lực lên motor
        self.apply_forces_to_motors(thrust, roll_moment, pitch_moment, yaw_moment)
        
        return {
            'altitude': current_altitude,
            'altitude_error': altitude_error,
            'thrust': thrust,
            'roll_moment': roll_moment,
            'pitch_moment': pitch_moment,
            'yaw_moment': yaw_moment
        }

# === HÀM KHỞI TẠO MÔI TRƯỜNG ===
def setup_simulation():
    """Khởi tạo môi trường simulation"""
    # Cấu hình simulation
    sim_config = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    simulation = SimulationContext(sim_config)
    simulation.set_camera_view(eye=[1.5, 0.5, 1.5], target=[0.0, 0.0, 1.0])
    
    # Thêm mặt đất và ánh sáng
    ground_config = sim_utils.GroundPlaneCfg()
    ground_config.func("/World/defaultGroundPlane", ground_config)
    
    light_config = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_config.func("/World/Light", light_config)
    
    # Khởi tạo robot
    robot_config = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_config = robot_config.replace(init_state=robot_config.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_config.spawn.func("/World/Crazyflie", robot_config.spawn, translation=robot_config.init_state.pos)
    
    robot = Articulation(robot_config)
    simulation.reset()
    
    return simulation, robot

# === HÀM MAIN ===
def main():
    print("[INFO] Initializing simulation...")
    
    # Khởi tạo simulation và robot
    simulation, robot = setup_simulation()
    
    # Lấy thông tin về motor
    propeller_body_ids = robot.find_bodies("m.*_prop")[0]
    allocation_matrix_inv = make_alloc_inv(device=simulation.device)
    
    # Khởi tạo bộ điều khiển
    controller = UAVAltitudeController(simulation, robot, propeller_body_ids, allocation_matrix_inv)
    
    # Khởi tạo đồ thị
    sim_dt = simulation.get_physics_dt()
    plotter = AltitudePlotter(DisplayConfig.WINDOW_SIZE, sim_dt)
    
    # Biến điều khiển vòng lặp
    simulation_time = 0.0
    step_count = 0
    log_interval = int(1.0 / sim_dt)  # Log mỗi giây
    
    print(f"[INFO] Target altitude = {ControlConfig.TARGET_ALTITUDE} m")
    print("[INFO] Starting simulation...")
    
    # Vòng lặp chính
    while simulation_app.is_running():
        # Bước điều khiển
        control_data = controller.control_step(ControlConfig.TARGET_ALTITUDE)
        
        # Cập nhật simulation
        robot.write_data_to_sim()
        simulation.step()
        simulation_time += sim_dt
        step_count += 1
        robot.update(sim_dt)
        
        # Log thông tin định kỳ
        if step_count % log_interval == 0:
            print(f"t={simulation_time:5.1f}s | "
                  f"altitude={control_data['altitude']:+.3f}m | "
                  f"error={control_data['altitude_error']:+.3f}m | "
                  f"thrust={control_data['thrust']:.4f}N")
        
        # Cập nhật đồ thị
        if step_count % DisplayConfig.PLOT_INTERVAL == 0:
            plotter.update(simulation_time, control_data['altitude'], ControlConfig.TARGET_ALTITUDE)
        
        # Cập nhật camera theo dõi UAV
        uav_position = robot.data.root_pos_w[0].cpu().numpy()
        simulation.set_camera_view(
            eye=(
                uav_position[0] + DisplayConfig.CAMERA_OFFSET[0],
                uav_position[1] + DisplayConfig.CAMERA_OFFSET[1],
                uav_position[2] + DisplayConfig.CAMERA_OFFSET[2]
            ),
            target=(uav_position[0], uav_position[1], uav_position[2]),
        )

if __name__ == "__main__":
    main()
    simulation_app.close()