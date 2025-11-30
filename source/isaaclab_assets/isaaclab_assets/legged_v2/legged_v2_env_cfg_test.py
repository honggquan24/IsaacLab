import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg, 
    ObservationGroupCfg, 
    ObservationTermCfg, 
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

from .legged_v2_cfg_test import LEGGED_ROBOT_V2_CFG_TEST

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from icecream import ic
from . import mdp

#./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Legged-Robot-V2-Balance --resume --load_run=2025-10-31_09-51-05 --checkpoint=model_350.pt --video

@configclass
class LeggedRobotV2SceneConfigTest(InteractiveSceneCfg):
    """Scene configuration for the legged robot environment."""
    num_envs: int = 1
    
    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Add terrain (commented out - using ground plane)
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=3,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="average",
    #         static_friction=0.05,
    #         dynamic_friction=0.05,
    #         restitution=0.0,
    #     ),
    #     debug_vis=True,          
    # )

    cfg_ground = AssetBaseCfg( 
        prim_path="/World/ground", 
        spawn=sim_utils.GroundPlaneCfg(), 
    )

    # Add robot 
    robot: Articulation = LEGGED_ROBOT_V2_CFG_TEST.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Add IMU sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v2/robot_legged_v2/Sensor",
        update_period=0.02,  # FIX: Changed from 0.1 to match control frequency (50Hz)
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=True,
    )

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v2/robot_legged_v2/Sensor",
    #     update_period=0.02,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(
    #         resolution=0.001,
    #         size=[0.02, 0.02],
    #     ),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )


@configclass
class ActionCfgTest:
    """Action configuration for joint effort control."""
    
    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "Left_Revolute_01",   # hip_left
            "Right_Revolute_01",  # hip_right 
            "Left_Revolute_02",   # knee_left
            "Right_Revolute_02",  # knee_right
            "Left_Revolute_03",   # ankle_left
            "Right_Revolute_03",  # ankle_right
            "Left_Revolute_04",   # wheel_left
            "Right_Revolute_04",  # wheel_right
        ],
        scale={
            "Left_Revolute_0[1-3]": 80.0,   # Leg joints
            "Right_Revolute_0[1-3]": 80.0,  # Leg joints
            "Left_Revolute_04": 100.0,      # Left wheel
            "Right_Revolute_04": 100.0,     # Right wheel
        },
        debug_vis=True,
    )


@configclass
class ObservationsCfgTest:
    """Observation configuration for the policy."""
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group."""
        
        # IMU sensors
        imu_lin_acc = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)
        
        # Joint states
        joint_pos = ObservationTermCfg(func=observations.joint_pos)
        joint_vel = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)
        
        # Previous actions (for smoothness)
        last_action = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Critic observation group - more comprehensive information."""
        
        # Root state (privileged information)
        root_pos_w = ObservationTermCfg(func=observations.root_pos_w)
        root_quat_w = ObservationTermCfg(func=observations.root_quat_w)
        root_lin_vel_w = ObservationTermCfg(func=observations.root_lin_vel_w)
        root_ang_vel_w = ObservationTermCfg(func=observations.root_ang_vel_w)
        
        # Base velocity (body frame)
        base_lin_vel = ObservationTermCfg(func=observations.base_lin_vel)
        
        # IMU data
        imu_lin_acc = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)
        
        # Joint states
        joint_pos = ObservationTermCfg(func=observations.joint_pos)
        joint_vel = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)
        
        # Previous actions
        last_action = ObservationTermCfg(func=observations.last_action)
        
        # Time information (for temporal learning)
        current_time = ObservationTermCfg(func=observations.current_time_s)
        remaining_time = ObservationTermCfg(func=observations.remaining_time_s)
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
        
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfgTest:
    """Event configuration for environment resets."""
    
    # Reset joints with small random offsets (avoid local minima)
    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (-0.05, 0.05), 
            "velocity_range": (-0.1, 0.1), 
        },
    )

    # Reset base with small noise (increase robustness)
    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x": (-0.1, 0.1),            
                "y": (-0.1, 0.1),            
                "z": (0.25, 0.25),       
                "roll": (-0.05, 0.05),         
                "pitch": (-0.05, 0.05),   
                "yaw": (-0.1, 0.1),  
            },
            "velocity_range": {
                "linear": (-0.05, 0.05),      
                "angular": (-0.05, 0.05),   
            },
        },
    )


@configclass
class RewardCfgTest:
    """Reward terms for the MDP."""
    
    # (1) Constant running reward - encourage survival
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=1.0
    )
    
    # (2) Failure penalty - penalize termination
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-2.0
    )
    
    # (3) Full RPY alignment (commented out - using pose alignment instead)
    rpy_alignment = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=1.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
        },
    )
    
    # (4) Pose alignment reward - encourage target joint configuration
    pose_alignment = RewardTermCfg(
        func=mdp.rewards.pose_align_reward,
        weight=4.0,
        params={
            # Uses default TARGET_JOINT_POS and JOINT_MASK from mdp.rewards
            "scale": 50.0,  # FIX: Explicitly set scale parameter
        },
    )
    
    # (5) Height reward when robot reach 0.4m
    # height = RewardTermCfg(
    #     func=mdp.rewards.height_reward,
    #     weight=5.0,
    #     params={
    #     },
    # )


@configclass
class TerminationsCfgTest:
    """Termination configuration for legged r

Có thể có trễ âm thanh khi chơi game hoặc xem video.
obot environment."""

    # TIME OUT - Episode ends after episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,  # Mark as timeout (not failure)
    )

    # ROOT HEIGHT BELOW MINIMUM (commented out for testing)
    # base_height = TerminationTermCfg(
    #     func=terminations.root_height_below_minimum,
    #     params={
    #         "minimum_height": 0.1,  # FIX: Changed from 0.01 (too low)
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    # BAD ORIENTATION (commented out for testing)
    # bad_orientation = TerminationTermCfg(
    #     func=terminations.bad_orientation,
    #     params={
    #         "limit_angle": math.pi / 3,  # FIX: Changed from pi/5 to pi/3 (60°)
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    # JOINT POSITION OUT OF SOFT LIMITS (commented out for testing)
    # joint_pos_limit = TerminationTermCfg(
    #     func=terminations.joint_pos_out_of_limit,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    # JOINT VELOCITY OUT OF LIMITS
    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 100.0,  # FIX: Changed from 200.0 (too high for safety)
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # JOINT EFFORT OUT OF LIMITS (commented out for testing)
    # joint_effort_limit = TerminationTermCfg(
    #     func=terminations.joint_effort_out_of_limit,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )


@configclass
class LeggedRobotV2EnvCfgTest(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""
    
    # Scene settings
    scene: LeggedRobotV2SceneConfigTest = LeggedRobotV2SceneConfigTest(
        num_envs=1,
        env_spacing=2.0,
    )

    # MDP components
    observations: ObservationsCfgTest = ObservationsCfgTest()
    actions: ActionCfgTest = ActionCfgTest()
    events: EventCfgTest = EventCfgTest()
    rewards: RewardCfgTest = RewardCfgTest()
    terminations: TerminationsCfgTest = TerminationsCfgTest()

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
        