"""Configuration for the UAV (Crazyflie quadcopter).

Robot: Crazyflie 2.x quadcopter
Structure:
  - body (chassis)
  - m1_joint, m2_joint, m3_joint, m4_joint (4 propeller joints)

Tasks:
  - Isaac-UAV-Hover: Hover at a target position

NOTE: USD file chưa có — tạm thời dùng CRAZYFLIE_CFG từ isaaclab_assets.robots.
      Khi có USD file riêng, thay bằng ArticulationCfg trỏ đến file USD đó:

      import os
      import isaaclab.sim as sim_utils
      from isaaclab.assets import ArticulationCfg
      CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
      UAV_USD_PATH = os.path.join(CURRENT_DIR, "usd_file", "uav.usd")
      UAV_CFG = ArticulationCfg(
          spawn=sim_utils.UsdFileCfg(usd_path=UAV_USD_PATH, ...),
          ...
      )
"""
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG  # isort:skip

# Tạm thời alias CRAZYFLIE_CFG — thay bằng ArticulationCfg riêng khi có USD file
UAV_CFG = CRAZYFLIE_CFG
"""Configuration for the UAV (Crazyflie) robot."""
