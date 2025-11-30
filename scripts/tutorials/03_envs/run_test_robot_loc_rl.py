# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive simulation with GUI controls for the legged robot.

Controls:
- GUI sliders: Adjust target joint positions in real-time
- Keyboard: 
  - SPACE: Pause/Resume simulation
  - R: Reset environment
  - P: Toggle position control mode
  - T: Toggle torque control mode
  - Arrow keys: Manual control (when in manual mode)
  - ESC: Exit

"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive legged robot simulation with GUI.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic behavior.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math
import carb

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
from isaaclab_assets import LeggedRobotV2EnvCfgTest
import omni
import omni.ui as ui
from pxr import Gf


class InteractiveSimulator:
    """Interactive simulator with GUI controls."""
    
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.robot = env.scene['robot']
        
        # Simulation state
        self.paused = False
        self.control_mode = "position"  # "position", "torque", "manual"
        self.step_count = 0
        self.print_interval = 100
        
        # Target joint positions (radians)
        self.target_joints = {
            "Left_Hip": 0.0,
            "Right_Hip": 0.0,
            "Left_Knee": math.radians(38.0),
            "Left_Ankle": math.radians(25.0),
            "Right_Knee": -math.radians(38.0),
            "Right_Ankle": -math.radians(25.0),
            "Left_Wheel": 0.0,
            "Right_Wheel": 0.0,
        }
        
        # Manual control velocities
        self.manual_velocities = torch.zeros(env.num_envs, 8, device=self.robot.device)
        
        # Subscribe to keyboard events
        self._setup_keyboard()
        
        # Create GUI
        self._create_gui()
        
        # Reset environment
        self.observations, self.extras = env.reset()
        
    def _setup_keyboard(self):
        """Setup keyboard event handlers."""
        appwindow = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, self._on_keyboard_event)
        
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Pause/Resume
            if event.input == carb.input.KeyboardInput.SPACE:
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RESUMED"
                print(f"\n[KEYBOARD] Simulation {status}\n")
                
            # Reset
            elif event.input == carb.input.KeyboardInput.R:
                print("\n[KEYBOARD] Resetting environment...\n")
                self.observations, self.extras = self.env.reset()
                self.step_count = 0
                
            # Toggle control modes
            elif event.input == carb.input.KeyboardInput.P:
                self.control_mode = "position"
                print(f"\n[KEYBOARD] Control mode: POSITION\n")
                
            elif event.input == carb.input.KeyboardInput.T:
                self.control_mode = "torque"
                print(f"\n[KEYBOARD] Control mode: TORQUE\n")
                
            elif event.input == carb.input.KeyboardInput.M:
                self.control_mode = "manual"
                print(f"\n[KEYBOARD] Control mode: MANUAL (Use arrow keys)\n")
                
            # Manual control (arrow keys)
            elif event.input == carb.input.KeyboardInput.UP:
                if self.control_mode == "manual":
                    self.manual_velocities[:, 2] += 0.1  # Left knee
                    self.manual_velocities[:, 4] += 0.1  # Right knee
                    
            elif event.input == carb.input.KeyboardInput.DOWN:
                if self.control_mode == "manual":
                    self.manual_velocities[:, 2] -= 0.1
                    self.manual_velocities[:, 4] -= 0.1
                    
            elif event.input == carb.input.KeyboardInput.LEFT:
                if self.control_mode == "manual":
                    self.manual_velocities[:, 6] -= 0.5  # Left wheel
                    
            elif event.input == carb.input.KeyboardInput.RIGHT:
                if self.control_mode == "manual":
                    self.manual_velocities[:, 7] += 0.5  # Right wheel
        
        return True
    
    def _create_gui(self):
        """Create interactive GUI window."""
        self.window = ui.Window("Robot Control Panel", width=400, height=600)
        
        with self.window.frame:
            with ui.VStack(spacing=10):
                # Title
                ui.Label("Legged Robot Interactive Control", 
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})
                
                ui.Spacer(height=10)
                
                # Control mode display
                with ui.HStack():
                    ui.Label("Control Mode:", width=120)
                    self.mode_label = ui.Label("Position", 
                                              style={"color": 0xFF00FF00})
                
                ui.Spacer(height=10)
                ui.Separator()
                ui.Spacer(height=10)
                
                # Joint sliders
                ui.Label("Target Joint Positions (degrees)", 
                        style={"font_size": 14})
                
                self.sliders = {}
                
                # Hip joints
                ui.Label("Hip Joints:", style={"font_size": 12})
                self.sliders["Left_Hip"] = self._create_slider(
                    "Left Hip", -45, 45, 0, self._update_joint)
                self.sliders["Right_Hip"] = self._create_slider(
                    "Right Hip", -45, 45, 0, self._update_joint)
                
                ui.Spacer(height=5)
                
                # Knee joints
                ui.Label("Knee Joints:", style={"font_size": 12})
                self.sliders["Left_Knee"] = self._create_slider(
                    "Left Knee", 0, 90, 38, self._update_joint)
                self.sliders["Right_Knee"] = self._create_slider(
                    "Right Knee", -90, 0, -38, self._update_joint)
                
                ui.Spacer(height=5)
                
                # Ankle joints
                ui.Label("Ankle Joints:", style={"font_size": 12})
                self.sliders["Left_Ankle"] = self._create_slider(
                    "Left Ankle", 0, 45, 25, self._update_joint)
                self.sliders["Right_Ankle"] = self._create_slider(
                    "Right Ankle", -45, 0, -25, self._update_joint)
                
                ui.Spacer(height=5)
                
                # Wheel joints
                ui.Label("Wheel Joints:", style={"font_size": 12})
                self.sliders["Left_Wheel"] = self._create_slider(
                    "Left Wheel", -180, 180, 0, self._update_joint)
                self.sliders["Right_Wheel"] = self._create_slider(
                    "Right Wheel", -180, 180, 0, self._update_joint)
                
                ui.Spacer(height=10)
                ui.Separator()
                ui.Spacer(height=10)
                
                # Action buttons
                with ui.HStack(spacing=10):
                    ui.Button("Reset to Default", clicked_fn=self._reset_sliders)
                    ui.Button("Reset Env", clicked_fn=self._reset_env)
                
                ui.Spacer(height=5)
                
                with ui.HStack(spacing=10):
                    ui.Button("Stand Pose", clicked_fn=self._set_stand_pose)
                    ui.Button("Crouch Pose", clicked_fn=self._set_crouch_pose)
                
                ui.Spacer(height=10)
                ui.Separator()
                ui.Spacer(height=10)
                
                # Statistics
                ui.Label("Statistics:", style={"font_size": 14})
                self.stats_labels = {}
                self.stats_labels["steps"] = ui.Label("Steps: 0")
                self.stats_labels["error"] = ui.Label("Pos Error: 0.0000 rad")
                self.stats_labels["reward"] = ui.Label("Total Reward: 0.0000")
                
    def _create_slider(self, name, min_val, max_val, default_val, callback):
        """Create a slider with label and value display."""
        with ui.HStack(height=30):
            ui.Label(f"{name}:", width=100)
            slider = ui.FloatSlider(
                min=min_val, 
                max=max_val, 
                width=150
            )
            value_label = ui.Label(f"{default_val:.1f}°", width=60)
            
            slider.model.set_value(default_val)
            
            def on_value_changed(model):
                value = model.get_value_as_float()
                value_label.text = f"{value:.1f}°"
                callback(name, value)
            
            slider.model.add_value_changed_fn(on_value_changed)
            
        return slider
    
    def _update_joint(self, joint_name, value_degrees):
        """Update target joint position from slider."""
        self.target_joints[joint_name] = math.radians(value_degrees)
    
    def _reset_sliders(self):
        """Reset all sliders to default values."""
        defaults = {
            "Left_Hip": 0, "Right_Hip": 0,
            "Left_Knee": 38, "Left_Ankle": 25,
            "Right_Knee": -38, "Right_Ankle": -25,
            "Left_Wheel": 0, "Right_Wheel": 0,
        }
        for name, value in defaults.items():
            self.sliders[name].model.set_value(value)
    
    def _reset_env(self):
        """Reset environment."""
        self.observations, self.extras = self.env.reset()
        self.step_count = 0
        print("\n[GUI] Environment reset\n")
    
    def _set_stand_pose(self):
        """Set robot to standing pose."""
        self.sliders["Left_Hip"].model.set_value(0)
        self.sliders["Right_Hip"].model.set_value(0)
        self.sliders["Left_Knee"].model.set_value(10)
        self.sliders["Right_Knee"].model.set_value(-10)
        self.sliders["Left_Ankle"].model.set_value(5)
        self.sliders["Right_Ankle"].model.set_value(-5)
    
    def _set_crouch_pose(self):
        """Set robot to crouching pose."""
        self.sliders["Left_Knee"].model.set_value(60)
        self.sliders["Right_Knee"].model.set_value(-60)
        self.sliders["Left_Ankle"].model.set_value(40)
        self.sliders["Right_Ankle"].model.set_value(-40)
    
    def _get_target_positions(self):
        """Get target positions as tensor in EXACT joint order:
        [0] Left_Revolute_01  (hip)
        [1] Right_Revolute_01 (hip)
        [2] Left_Revolute_02  (knee)
        [3] Left_Revolute_03  (ankle)
        [4] Right_Revolute_02 (knee)
        [5] Right_Revolute_03 (ankle)
        [6] Left_Revolute_05  (passive) → giữ nguyên 0.0
        [7] Right_Revolute_05 (passive) → giữ nguyên 0.0
        [8] Left_Revolute_06  (passive) → giữ nguyên 0.0
        [9] Right_Revolute_06 (passive) → giữ nguyên 0.0
        [10] Left_Revolute_04 (wheel)
        [11] Right_Revolute_04 (wheel)
        """
        target = torch.tensor([
            self.target_joints["Left_Hip"],     # 0
            self.target_joints["Right_Hip"],    # 1
            self.target_joints["Left_Knee"],    # 2
            self.target_joints["Left_Ankle"],   # 3
            self.target_joints["Right_Knee"],   # 4
            self.target_joints["Right_Ankle"],  # 5
            0.0,  # 6: Left_Revolute_05 (passive)
            0.0,  # 7: Right_Revolute_05 (passive)
            0.0,  # 8: Left_Revolute_06 (passive)
            0.0,  # 9: Right_Revolute_06 (passive)
            self.target_joints["Left_Wheel"],   # 10
            self.target_joints["Right_Wheel"],  # 11
        ], dtype=torch.float32, device=self.robot.device)
        
        return target.unsqueeze(0).expand(self.env.num_envs, -1)
    
    def _update_statistics(self):
        """Update GUI statistics."""
        current_pos = self.robot.data.joint_pos
        target_pos = self._get_target_positions()
        error = torch.norm(current_pos - target_pos, dim=-1).mean().item()
        
        # Get total reward from extras
        log = self.extras.get('log', {})
        total_reward = sum([
            v[0].item() if v.dim() > 0 else v.item()
            for k, v in log.items() 
            if 'Reward' in k and torch.is_tensor(v)
        ])
        
        self.stats_labels["steps"].text = f"Steps: {self.step_count}"
        self.stats_labels["error"].text = f"Pos Error: {error:.4f} rad"
        self.stats_labels["reward"].text = f"Total Reward: {total_reward:.4f}"
        
        # Update control mode label
        self.mode_label.text = self.control_mode.upper()
    
    def step(self):
        """Perform one simulation step."""
        if self.paused:
            # Still step simulation for rendering, but skip control
            self.env.sim.step(render=True)
            return

        current_time = self.env.sim.current_time

        if self.control_mode == "position":
            # ───────────────────────────────────────────────────────
            # ✅ POSITION CONTROL: Set joint positions DIRECTLY (bypass controller)
            #    Vì stiffness=0 → không có PID → phải dùng set_joint_position()
            # ───────────────────────────────────────────────────────
            target_pos = self._get_target_positions()  # [num_envs, 12]
            
            # Gửi trực tiếp vào simulation (instantaneous)
            self.robot.set_joint_position_target(target_pos)
            self.robot.write_data_to_sim()  # 🔑 BẮT BUỘC
            
            # Step simulation (với action=0 vì đã set position rồi)
            actions = torch.zeros_like(self.env.action_manager.action)
            self.observations, rewards, terminated, truncated, self.extras = \
                self.env.step(actions)

        elif self.control_mode == "torque":
            # ───────────────────────────────────────────────────────
            # ✅ TORQUE CONTROL: Tính torque bằng PD từ target positions
            # ───────────────────────────────────────────────────────
            target_pos = self._get_target_positions()
            current_pos = self.robot.data.joint_pos
            current_vel = self.robot.data.joint_vel

            # PD gains — điều chỉnh theo robot của bạn
            kp = 50.0
            kd = 1.0
            
            # Chỉ điều khiển 8 khớp actuated (0-5, 10-11), bỏ qua passive (6-9)
            torque = torch.zeros_like(current_pos)
            torque[:, [0,1,2,3,4,5,10,11]] = (
                kp * (target_pos[:, [0,1,2,3,4,5,10,11]] - current_pos[:, [0,1,2,3,4,5,10,11]])
                - kd * current_vel[:, [0,1,2,3,4,5,10,11]]
            )
            
            # Giới hạn torque theo effort_limit trong config (±1000)
            torque = torch.clamp(torque, -1000.0, 1000.0)
            
            # Chỉ truyền 8 action dimensions (theo ActionCfgTest.joint_names)
            actions = torque[:, [0,1,2,3,4,5,10,11]]  # ← ĐÚNG THỨ TỰ ACTION!
            self.observations, rewards, terminated, truncated, self.extras = \
                self.env.step(actions)

        elif self.control_mode == "manual":
            # ───────────────────────────────────────────────────────
            # ✅ MANUAL CONTROL: Dùng self.manual_velocities như effort
            # ───────────────────────────────────────────────────────
            # Decay nhẹ để tránh tích lũy
            self.manual_velocities *= 0.98
            actions = self.manual_velocities.clone()
            self.observations, rewards, terminated, truncated, self.extras = \
                self.env.step(actions)

        # ───────────────────────────────────────────────────────────
        # Cập nhật thống kê & debug
        # ───────────────────────────────────────────────────────────
        self.step_count += 1
        
        # Cập nhật GUI mỗi 5 bước
        if self.step_count % 5 == 0:
            self._update_statistics()
        
        # In log mỗi 100 bước
        if self.step_count % self.print_interval == 0:
            target_pos = self._get_target_positions()
            current_pos = self.robot.data.joint_pos
            error = torch.norm(current_pos - target_pos, dim=-1).mean().item()
            print(f"[{current_time:.2f}s | Step {self.step_count}] "
                f"Mode: {self.control_mode.upper():8} | "
                f"Joint error: {error:.4f} rad")

        # Xử lý tự reset (terminated/truncated)
        if terminated.any() or truncated.any():
            reset_env_ids = torch.nonzero(terminated | truncated).flatten()
            print(f"[INFO] Auto-reset envs: {reset_env_ids.tolist()}")
            # → Isaac Lab tự xử lý reset, không cần gọi lại reset()


def main():
    """Main function."""
    # Create environment
    env_cfg = LeggedRobotV2EnvCfgTest()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.seed(args_cli.seed)
    
    # Create interactive simulator
    sim = InteractiveSimulator(env)
    
    print("\n" + "="*70)
    print("INTERACTIVE SIMULATION")
    print("="*70)
    print("\nKeyboard Controls:")
    print("  SPACE    - Pause/Resume simulation")
    print("  R        - Reset environment")
    print("  P        - Position control mode")
    print("  T        - Torque control mode")
    print("  M        - Manual control mode")
    print("  Arrow    - Manual control (in manual mode)")
    print("  ESC      - Exit")
    print("\nGUI Controls:")
    print("  - Use sliders to adjust target joint positions")
    print("  - Click buttons for preset poses")
    print("="*70 + "\n")
    
    # Run simulation loop
    while simulation_app.is_running():
        sim.step()
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()