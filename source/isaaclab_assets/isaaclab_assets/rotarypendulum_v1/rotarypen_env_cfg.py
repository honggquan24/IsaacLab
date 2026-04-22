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

from .rotarypen_cfg import ROTARY_ROBOT_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from icecream import ic
import isaaclab.envs.mdp as mdp
from .mdp.terminations import *
from .mdp.rewards import *

@configclass
class RotarypenSceneConfig(InteractiveSceneCfg):
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
    robot: Articulation = ROTARY_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


@configclass
class ActionsCfg :
    # joint_effort = mdp.actions.actions_cfg.JointEffortActionCfg (joint_names=["Slider_1"],asset_name="robot",scale=1.0)
    joint_effort = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "Revolute_1",
        ],
        scale={
            "Revolute_1": 25.0, 
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
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for Critic group.""" 
        joint_pos = ObsTerm(func=mdp.joint_pos,params={"asset_cfg": SceneEntityCfg("robot")},)
        joint_vel = ObsTerm(func=mdp.joint_vel,params={"asset_cfg": SceneEntityCfg("robot")},)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # on reset
    reset_pole_position = EventTerm(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_[1-2]"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )
        
@configclass
class RewardCfg:
    """Reward terms for the MDP."""
    
    # (1) Constant running reward - encourage survival
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=0.5
    )
    # (2) Failure penalty - penalize termination0
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-10.0
    )
    reward_angle = RewardTermCfg (
        func = reward_angle_pen,
        weight = 6.0
    )
    penalty_vel_for_pen = RewardTermCfg (
        func = penalty_vel_pen,
        weight = 1.0
    )
    penalty_vel_for_rotary = RewardTermCfg (
        func = penalty_vel_Rotary,
        weight = 2.0
    )
    bonus = RewardTermCfg (
        func = Reward_bonus_near,
        weight = 1.0
    )

    reward_when_center_rotary = RewardTermCfg (
        func = penalty_when_center,
        weight = 3.0
    )

    

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
    # joint_limit = TerminationTermCfg (
    #     func = reset_joint_limit,
    # )

@configclass
class RotarypenEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""
    
    # Scene settings
    scene: RotarypenSceneConfig = RotarypenSceneConfig(
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
        self.episode_length_s = 30  # Episode duration
        
        # Viewer settings
        self.viewer.eye = (0.0, 5.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point
                
        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps

