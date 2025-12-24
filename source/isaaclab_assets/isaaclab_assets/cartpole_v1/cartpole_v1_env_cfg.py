import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm, 
    ObservationGroupCfg as ObsGroup, 
    ObservationTermCfg as ObsTerm, 
    RewardTermCfg as RewTerm,
    SceneEntityCfg, 
    TerminationTermCfg as DoneTerm,
    CommandTermCfg
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .cartpole_v1_cfg import CARTPOLE_ROBOT_CFG
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.envs.mdp as mdp
from .mdp.rewards import *

@configclass
class CartpoleRobotV1SceneConfig(InteractiveSceneCfg):
    """Scene configuration for the cartpole environment."""
    num_envs: int = 1024  # Tăng lên để training nhanh hơn
    
    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    cfg_ground = AssetBaseCfg( 
        prim_path="/World/ground", 
        spawn=sim_utils.GroundPlaneCfg(), 
    )

    # Add robot 
    robot: Articulation = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    # Sửa tên command từ "pose_command" thành "cart_command" để nhất quán
    cart_command = mdp.UniformPoseCommandCfg(  # Dùng UniformPoseCommandCfg thay vì UniformPose2dCommandCfg
        asset_name="robot",
        body_name="cart",  # QUAN TRỌNG: Tên body trong USD file của cart
        make_quat_unique=False,  # Không cần unique quaternion cho CartPole
        
        # Ranges cho position và orientation
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),      # Di chuyển trong khoảng 2m
            pos_y=(-0.5, 0.5),       # Không di chuyển theo trục Y
            pos_z=(0.0, 0.0),       # Luôn ở độ cao 0
            roll=(0.0, 0.0),        # Không roll
            pitch=(0.0, 0.0),       # Không pitch
            yaw=(0.0, 0.0),         # Không yaw (cart chỉ đi thẳng)
        ),
        
        # Thời gian thay đổi command
        resampling_time_range=(5.0, 5.0),  # 5-8 giây đổi command 1 lần
        
        # Visualization
        debug_vis=True,  # Hiển thị marker cho debug
    )

@configclass
class ActionsCfg:
    """Action terms for the MDP."""
    
    joint_effort = mdp.actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart", "cart_to_pole"],  # Chỉ điều khiển joint cart
        scale={
            "slider_to_cart": 100.0,  # Scale force
            "cart_to_pole": 0.0
        },
        debug_vis=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # === CÁC OBS CƠ BẢN ===
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart", "cart_to_pole"])},
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart", "cart_to_pole"])},
        )
        
        # 1. Lấy command trực tiếp (target position)
        cart_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "cart_command"},  # Phải trùng với tên trong CommandsCfg
        )
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()




@configclass
class EventCfg:
    """Configuration for events."""
    
    # Reset cart position
    reset_cart_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )
    
    # Reset pole position
    reset_pole_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )

class RewardsCfg:
    """Reward terms for the MDP using only the new functions."""
    
    # === BASIC REWARDS (vẫn cần để training ổn định) ===
    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0
    )
    
    terminating = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0
    )
    
    # Reward cho việc tracking command position
    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=10.0,  
        params={
            "std": 0.02,  
            "command_name": "cart_command",  
        },
    )
    
    pole_upright = RewTerm(
        func=joint_pos_target_l2,
        weight=-0.5,  
        params={
            "target": 0.0,  # Target angle = 0 (thẳng đứng)
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
        },
    )
    
    cart_center = RewTerm(
        func=joint_pos_target_l2,
        weight=-0.1,  
        params={
            "target": 0.0,  
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    # TIME OUT - Episode ends after episode_length_s
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,  # Mark as timeout (not failure)
    )
    
@configclass
class CartPoleV1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""
    
    # Scene settings
    scene: CartpoleRobotV1SceneConfig = CartpoleRobotV1SceneConfig(
        num_envs=1024,  # Tăng số env để training nhanh
        env_spacing=4.0,
    )
    
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 1
        self.episode_length_s = 10
        
        # Viewer settings
        self.viewer.eye = (0.0, 5.0, 10.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # Nhìn vào cart
        
        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps
        
        # Optimize for training
        self.sim.device = "cuda"  
