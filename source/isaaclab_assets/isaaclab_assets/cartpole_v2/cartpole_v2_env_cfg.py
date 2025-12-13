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
    TerminationTermCfg
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import (
    CameraCfg,
    ContactSensorCfg,
    RayCasterCfg,
    ImuCfg,
    patterns
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from .cartpole_v2_cfg import CARTPOLE_V2_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from icecream import ic
import isaaclab.envs.mdp as mdp
from .mdp.rewards import *
from .mdp.terminations import *

@configclass
class CartpoleRobotV2SceneConfig(InteractiveSceneCfg):
    """Scene configuration for the legged robot environment."""
    num_envs: int = 1
    
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
    robot: Articulation = CARTPOLE_V2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

@configclass
class ActionsCfg :
    # joint_effort = mdp.actions.actions_cfg.JointEffortActionCfg (joint_names=["Slider_1"],asset_name="robot",scale=1.0)
    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "Slider_1",
            # "Revolute_1"ManagerBasedRLEnvCfg,
            # "Revolute_2"
        ],
        scale={
            "Slider_1": 100.0, 
            # "Revolute_1": 0.0,
            # "Revolute_2": 0.0,
        },
        debug_vis=True,
    )
@configclass
class ObservationsCfg:

    """Observation specifications for the environment."""
    @configclass

    class PolicyCfg(ObsGroup):

        """Observations for policy group."""
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos,params={"asset_cfg": SceneEntityCfg("robot")},)
        joint_vel = ObsTerm(func=mdp.joint_vel,params={"asset_cfg": SceneEntityCfg("robot")},)
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    # on reset
    reset_cart_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Slider_1"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_[1,2]"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )

@configclass
class RewardCfg:
    alive = RewardTermCfg(func=rewards.is_alive, weight=0.2)
    
    terminating = RewardTermCfg(func=rewards.is_terminated, weight=-5.0)

    # rewards_rv1 = RewardTermCfg(
    #     func=cartpole_reward_joint_pos_rv1,
    #     weight=2.0)
    
    rewards_rv2 = RewardTermCfg(
        func=cartpole_reward_joint_pos_rv2_rv1, 
        weight=5.0)
    
    penalty_vel = RewardTermCfg(
        func=cartpole_penalty_joint_vel, 
        weight=-0.5)
    
    penalty_vel_p1 = RewardTermCfg(
        func=cartpole_penalty_joint_vel_pe1, 
        weight=-1.2)
    
    penalty_vel_p2 = RewardTermCfg(
        func=cartpole_penalty_joint_vel_pe2, 
        weight=-1.2)
    
    # penalty_fall_p1 = RewardTermCfg(
    #     func=cartpole_penalty_fall_p1, 
    #     weight=-2.5)
      
    # penalty_fall_p2 = RewardTermCfg(
    #     func=cartpole_penalty_fall_p2, 
    #     weight=-3.7)
    
    reward_cart = RewardTermCfg(
        func=cart_center_reward, 
        weight=0.2)
    
    penalty_cart = RewardTermCfg(
        func=cart_not_center_penalty, 
        weight=-1.0)



    




@configclass
class TerminationsCfg:
    """Termination configuration for legged r

Có thể có trễ âm thanh khi chơi game hoặc xem video.
obot environment."""

    # TIME OUT - Episode ends after episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,  # Mark as timeout (not failure)
    )
    # reset_cartpole1 = TerminationTermCfg(
    #     func = Cart_pole_angle_reset,
    # )

    # reset_cartpole1 = TerminationTermCfg(
    #     func = Cart_pole_angle_reset_1,
    # )

    # reset_cartpole2 = TerminationTermCfg(
    #     func = Cart_pole_pos_reset,
    # )
    # cart_out = TerminationTermCfg(
    #     func=cartpole_terminate_cart_out,
    #     params={"x_limit": 0.99},
    # )

@configclass
class CartPoleV2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""
    
    # Scene settings
    scene: CartpoleRobotV2SceneConfig = CartpoleRobotV2SceneConfig(
        num_envs=1,
        env_spacing=2.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2  # Control freq = 60/2 = 30 Hz
        self.episode_length_s = 100  # Episode duration
        
        # Viewer settings
        self.viewer.eye = (0.0, 5.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point
                
        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps


