import math
import os
import torch
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg
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

# from .cartpole_v2_cfg import CARTPOLE_V2_CFG
from .mobierobotv1_cfg import MOBIE_ROBOT_CFG
from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from icecream import ic
import isaaclab.envs.mdp as mdp

from .mdp.rewards import *
from .mdp.terminations import *
from .mdp.observations import *


from isaaclab.envs.mdp.commands.commands_cfg import *

current_dir = os.path.dirname(os.path.abspath(__file__))
track = os.path.join(
    current_dir, "usd_file", "trackv1.usd"
)

map2 = os.path.join(
    current_dir, "usd_file", "map2.usd"
)


@configclass
class MobieRobotV1SceneConfig(InteractiveSceneCfg):
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

    track = AssetBaseCfg(
        prim_path="/World/track",
        spawn=sim_utils.UsdFileCfg(
            usd_path=track,
        ),
    )
    # map2 = AssetBaseCfg(
    #     prim_path="/World/map2",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=map2,
    #     ),
    # )

    # Add robot 
    robot: Articulation = MOBIE_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    # Add sensor
    imu= ImuCfg(
        prim_path="/World/envs/env_.*/Robot/Mobie_robot_RL/base_link",
        offset=ImuCfg.OffsetCfg(
        pos=(0.0, 0.0, -0.35),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    update_period=0.0,
    debug_vis=False,
    )

    raycaster = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Mobie_robot_RL/base_link",
        mesh_prim_paths=["/World/track"],
        update_period=0.02,
        offset= RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.002),
            rot = (0.7071, 0.0, 0.0, 0.7071),
        ),

        ray_alignment="base",
        pattern_cfg=LidarPatternCfg(
            channels=1,
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360/6,
            vertical_fov_range=(-1.0, 1000.0),
        ),

        max_distance=10.0,
        drift_range=(-0.0, 0.0),
        ray_cast_drift_range={
            "x":(-0.01,0.01),
            "y":(-0.01,0.01),
            "z":(0,0)
            },
        debug_vis=False,
    )

@configclass
class ActionsCfg :
    joint_effort = actions.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "Revolute1",
            "Revolute2"
        ],
        scale={
            "Revolute1": 100.0,
            "Revolute2": 100.0,
        },
        debug_vis=True,
        
    )


@configclass

class CommandCfg:
    goal_pos = UniformPose2dCommandCfg(
        resampling_time_range= (10.0,10.0),
        debug_vis= False,
        asset_name= 'robot',
        simple_heading=False,
        ranges = UniformPose2dCommandCfg.Ranges(
            pos_x=(-4.5, 4.5),
            pos_y=(-4.5, 4.5),
            heading=(0.0 , 3.14), )
        )


@configclass
class ObservationsCfg:

    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):

        """Observations for policy group."""
        # observation terms (order preserved)
        joint_vel = ObsTerm(func=mdp.joint_vel,params={"asset_cfg": SceneEntityCfg("robot")},)
        imu_linear_vel_ob = ObsTerm (func = imu_linear_vel,params={"asset_cfg":SceneEntityCfg("imu")},)
        imu_ang_vel_ob = ObsTerm (func = imu_ang_vel_z,params={"asset_cfg":SceneEntityCfg("imu")},)
        raycaster_obs = ObsTerm (func = raycaster_observation, params={"asset_cfg":SceneEntityCfg("raycaster")},)
        # goal_command = ObsTerm(func=command_distance, params={"asset_cfg": SceneEntityCfg("robot")},)
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # observation groups
    @configclass  
    class CriticCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        joint_vel = ObsTerm(func=mdp.joint_vel,params={"asset_cfg": SceneEntityCfg("robot")},)
        imu_linear_vel_ob = ObsTerm (func = imu_linear_vel,params={"asset_cfg":SceneEntityCfg("imu")},)
        imu_ang_vel_ob = ObsTerm (func = imu_ang_vel_z,params={"asset_cfg":SceneEntityCfg("imu")},)
        raycaster_obs = ObsTerm (func = raycaster_observation, params={"asset_cfg":SceneEntityCfg("raycaster")},)
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute[1-2]"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )
    reset_root_state = EventTerm(
        func=mdp.events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
        },
)

@configclass
class RewardCfg:

    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=1.0
    )
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-500.0
    )
    rewards_linear_velocity = RewardTermCfg(
        func=reward_linear_velocity,
        weight=10.0 
    )

    rewards_forward = RewardTermCfg(
        func=reward_when_reward_forward,
        weight=0.03
    )

    rewards_avoid_obstacle = RewardTermCfg(
        func=reward_avoid_obstacle,
        weight=3.0,
    )
    penalty_spin_when_stuck_robot = RewardTermCfg(
        func=penalty_spin_when_stuck,
        weight=0.01, 
    )

    reward_raycast3 = RewardTermCfg(
        func=raycast3_forward,
        weight=1.0,
    )

    reward_raycast0 = RewardTermCfg(
        func=raycast0_forward,
        weight=1.0,
    )

    # reward_goal = RewardTermCfg(
    #     func= reward_when_at_goal,
    #     weight=6.0, 
    # )


@configclass
class TerminationsCfg:
    # TIME OUT - Episode ends after episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,  # Mark as timeout (not failure)
    )

    collision = TerminationTermCfg(
        func=termination_on_collision,
        params={
            "asset_cfg": SceneEntityCfg("raycaster")
    },
        time_out=False, 
)

    
@configclass
class MobieRobotV1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""  
    # Scene settings
    scene: MobieRobotV1SceneConfig = MobieRobotV1SceneConfig(
        num_envs=1,
        env_spacing=0.0,
    )
    commands: CommandCfg = CommandCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()


    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2  # Control freq = 60/2 = 30 Hz
        self.episode_length_s = 50  # Episode duration
        
        # Viewer settings
        self.viewer.eye = (0.0, 5.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point
                
        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps



#./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py  --task Isaac-MobieRobot-V1-Run


