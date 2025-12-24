import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm, 
    ObservationGroupCfg as ObsGroup, 
    ObservationTermCfg as ObsTerm, 
    RewardTermCfg,
    SceneEntityCfg, 
    TerminationTermCfg,
    CommandTermCfg as CommandTerm  # <-- THÊM DÒNG NÀY
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from .cartpole_v1_cfg import CARTPOLE_ROBOT_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from icecream import ic
import isaaclab.envs.mdp as mdp

from .mdp.rewards import *


@configclass
class CartPoseCommandCfg(CommandTerm):
    """Command cho CartPole: Điều khiển vị trí xe (cart)"""
    
    class_type: type = "CartPoseCommand"  # Sẽ tạo class riêng
    
    asset_name: str = "robot"  # Tên robot trong scene
    joint_name: str = "Slider_1"  # Joint cần điều khiển
    
    # Thời gian thay đổi command (giây)
    resampling_time_range: tuple[float, float] = (3.0, 5.0)
    
    @configclass
    class Ranges:
        """Phạm vi giá trị cho command"""
        # Vị trí mục tiêu cho xe (-2m đến 2m)
        target_pos: tuple[float, float] = (-1.5, 1.5)
        
        # Có thể thêm velocity command nếu muốn
        # target_vel: tuple[float, float] = (-0.5, 0.5)
    
    ranges: Ranges = Ranges()
    
    # Visualization marker cho command
    goal_marker_cfg = sim_utils.CylinderCfg(
        radius=0.05,
        height=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )

@configclass
class PoleAngleCommandCfg(CommandTerm):
    """Command cho CartPole: Điều khiển góc của pole"""
    
    class_type: type = "PoleAngleCommand"
    
    asset_name: str = "robot"
    joint_name: str = "Revolute_1"  # Joint của pole
    
    resampling_time_range: tuple[float, float] = (2.0, 4.0)
    
    @configclass
    class Ranges:
        """Phạm vi góc cho pole (radian)"""
        # Góc mục tiêu (-30° đến 30° từ vertical)
        target_angle: tuple[float, float] = (-0.5, 0.5)  # ~ -30° to 30°
    
    ranges: Ranges = Ranges()
    
    goal_marker_cfg = sim_utils.ConeCfg(
        radius=0.05,
        height=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )

@configclass
class CompositeCommandCfg(CommandTerm):
    """Command kết hợp cả vị trí xe và góc pole"""
    
    class_type: type = "CompositeCartPoleCommand"
    
    asset_name: str = "robot"
    
    resampling_time_range: tuple[float, float] = (4.0, 6.0)
    
    @configclass
    class Ranges:
        """Phạm vi cho cả position và angle"""
        cart_pos: tuple[float, float] = (-1.0, 1.0)
        pole_angle: tuple[float, float] = (-0.3, 0.3)
    
    ranges: Ranges = Ranges()

@configclass
class CommandsCfg:
    """Tổng hợp tất cả commands cho CartPole"""
    
    # Chọn 1 trong 3 loại command:
    
    # Option 1: Chỉ điều khiển vị trí xe
    cart_position: CartPoseCommandCfg = CartPoseCommandCfg()
    
    # Option 2: Chỉ điều khiển góc pole
    # pole_angle: PoleAngleCommandCfg = PoleAngleCommandCfg()
    
    # Option 3: Điều khiển cả hai
    # composite: CompositeCommandCfg = CompositeCommandCfg()
