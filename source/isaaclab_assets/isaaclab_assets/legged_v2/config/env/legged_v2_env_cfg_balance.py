import math
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
    ContactSensorCfg,
    RayCasterCfg,
    ImuCfg,
    patterns
)
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from ..robot.legged_v2_cfg import LEGGED_ROBOT_V2_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
from ... import mdp
from isaaclab.envs.mdp import *



@configclass
class LeggedRobotV2SceneConfig(InteractiveSceneCfg):
    """Scene configuration for the legged robot environment."""
    num_envs: int = 1
    
    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Add terrain (commented out - using ground plane)
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/Ground",
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
        prim_path="/World/Ground", 
        spawn=sim_utils.GroundPlaneCfg(), 
    )

    # Add robot 
    robot: Articulation = LEGGED_ROBOT_V2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Add IMU sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/.*/Sensor",
        update_period=0.02,  # FIX: Changed from 0.1 to match control frequency (50Hz)
        gravity_bias=(0.0, 0.0, 0.0),
        # debug_vis=True,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/.*/Sensor",
        update_period=0.01,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[0.3, 0.3],
        ),
        # debug_vis=True,
        mesh_prim_paths=["/World/Ground"],
    )
    
    contact_forces_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/.*/Left_Leg", 
        update_period=0.01, 
        # debug_vis=True
    )

    contact_forces_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/.*/Right_Leg", 
        update_period=0.01, 
        # debug_vis=True
    )
    
    contact_forces_wheel_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/.*/Wheel", 
        update_period=0.01, 
        # debug_vis=True
    )

    contact_forces_wheel_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*/.*/Wheel_01", 
        update_period=0.01, 
        # debug_vis=True
    )


@configclass
class ActionCfg:
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
        scale=200.0,
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
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
        
        # Pose
        body_pose_w = ObservationTermCfg(func=observations.body_pose_w)
        
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
class EventCfg:
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
class RewardCfg:
    """Reward terms for balance task."""

    # (1) Survival reward
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=10.0,
    )

    # (2) Termination penalty
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-200.0,
    )

    # (3) IMU alignment (trọng tâm)
    rpy_alignment = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=7.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
            "tolerance": 0.1,
        },
    )

    # (4) Joint target reward - đứng thẳng
    # Left Leg 
    # left_hip = RewardTermCfg(
    #     func=mdp.rewards.joint_pos_target_l2,
    #     weight=-1000.0,
    #     params={
    #         "target": 0.0,
    #         "asset_cfg": SceneEntityCfg(
    #             name="robot",
    #             joint_names=["Left_Revolute_01"]
    #         ),
    #     },
    # )

    # left_knee = RewardTermCfg(
    #     func=mdp.rewards.joint_pos_target_l2,
    #     weight=-12.0,
    #     params={
    #         "target": 0.0,
    #         "asset_cfg": SceneEntityCfg(
    #             name="robot",
    #             joint_names=["Left_Revolute_02"]
    #         ),
    #     },
    # )

    # left_ankle = RewardTermCfg(
    #     func=mdp.rewards.joint_pos_target_l2,
    #     weight=-12.0,
    #     params={
    #         "target": 0.0,
    #         "asset_cfg": SceneEntityCfg(
    #             name="robot",
    #             joint_names=["Left_Revolute_03"]
    #         ),
    #     },
    # )

    # # Right Leg 
    # right_hip = RewardTermCfg(
    #     func=mdp.rewards.joint_pos_target_l2,
    #     weight=-1000.0,
    #     params={
    #         "target": 0.0,
    #         "asset_cfg": SceneEntityCfg(
    #             name="robot",
    #             joint_names=["Right_Revolute_01"]
    #         ),
    #     },
    # )

    # right_knee = RewardTermCfg(
    #     func=mdp.rewards.joint_pos_target_l2,
    #     weight=-12.0,
    #     params={
    #         "target": 0.0,
    #         "asset_cfg": SceneEntityCfg(
    #             name="robot",
    #             joint_names=["Right_Revolute_02"]
    #         ),
    #     },
    # )

    # right_ankle = RewardTermCfg(
    #     func=mdp.rewards.joint_pos_target_l2,
    #     weight=-12.0,
    #     params={
    #         "target": 0.0,
    #         "asset_cfg": SceneEntityCfg(
    #             name="robot",
    #             joint_names=["Right_Revolute_03"]
    #         ),
    #     },
    # )

    # # (5) Height reward
    # height = RewardTermCfg(
    #     func=mdp.rewards.height_reward,
    #     weight=6.0,
    #     params={
    #         "target_height": 0.5,
    #     },
    # )

    # # (6) Angular velocity stability
    # ang_vel = RewardTermCfg(
    #     func=mdp.rewards.angular_velocity_reward,
    #     weight=4.0,
    #     params={
    #         "target_angular_vel": 0.0,
    #     },
    # )

    # # (7) Linear velocity stability (đứng yên)
    # lin_vel = RewardTermCfg(
    #     func=mdp.rewards.linear_velocity_reward,
    #     weight=4.0,
    #     params={
    #         "target_linear_vel": 0.0,
    #     },
    # )

    # # (8) Balanced torque between legs
    # # Hip balance
    # balance_hip = RewardTermCfg(
    #     func=mdp.rewards.joint_force_balance,
    #     weight=-0.005,
    #     params={
    #         "left_cfg": SceneEntityCfg(name="robot", joint_names=["Left_Revolute_01"]),
    #         "right_cfg": SceneEntityCfg(name="robot", joint_names=["Right_Revolute_01"]),
    #     },
    # )

    # # Knee balance
    # balance_knee = RewardTermCfg(
    #     func=mdp.rewards.joint_force_balance,
    #     weight=-0.005,
    #     params={
    #         "left_cfg": SceneEntityCfg(name="robot", joint_names=["Left_Revolute_02"]),
    #         "right_cfg": SceneEntityCfg(name="robot", joint_names=["Right_Revolute_02"]),
    #     },
    # )

    # # Ankle balance
    # balance_ankle = RewardTermCfg(
    #     func=mdp.rewards.joint_force_balance,
    #     weight=-0.005,
    #     params={
    #         "left_cfg": SceneEntityCfg(name="robot", joint_names=["Left_Revolute_03"]),
    #         "right_cfg": SceneEntityCfg(name="robot", joint_names=["Right_Revolute_03"]),
    #     },
    # )

    
    # (9) Action smoothness penalty
    action_rate = RewardTermCfg(
        func=action_rate_l2,
        weight=-2.0,
    )

    # (10) Joint acceleration — very small penalty
    joint_accel = RewardTermCfg(
        func=joint_acc_l2,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (11) Joint velocity penalty
    joint_vel = RewardTermCfg(
        func=joint_vel_l2,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (12) Vertical linear velocity penalty
    lin_vel_z = RewardTermCfg(
        func=lin_vel_z_l2,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (13) Angular velocity XY penalty
    ang_vel_xy = RewardTermCfg(
        func=ang_vel_xy_l2,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (14) Flat orientation penalty
    flat_orientation = RewardTermCfg(
        func=flat_orientation_l2,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (15) Base height penalty
    base_height = RewardTermCfg(
        func=base_height_l2,
        weight=-1e-5,
        params={
            "target_height": 0.5,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (16) Body linear acceleration penalty
    body_lin_acc = RewardTermCfg(
        func=body_lin_acc_l2,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (17) Joint velocity limits soft penalty
    joint_vel_limits = RewardTermCfg(
        func=joint_vel_limits,
        weight=-1e-5,
        params={
            "soft_ratio": 0.1,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # (18) Torque limits soft penalty
    applied_torque_limits = RewardTermCfg(
        func=applied_torque_limits,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (19) Contact forces penalty
    contact_forces = RewardTermCfg(
        func=contact_forces,
        weight=-1e-5,
        params={
            "threshold": 200.0,
            "sensor_cfg": SceneEntityCfg(name="contact_forces_left"),
        },
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

    # ROOT HEIGHT BELOW MINIMUM (commented out for testing)
    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.25,  # 
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # BAD ORIENTATION (commented out for testing)
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 3,  # FIX: Changed from pi/5 to pi/3 (60°)
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # JOINT VELOCITY OUT OF LIMITS
    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 250.0,  # FIX: Changed from 200.0 (too high for safety)
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # # Contact effort 
    contact_left = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 200.0, 
            "sensor_cfg": SceneEntityCfg(name="contact_forces_left") 
        }
    ) 
    
    contact_right = TerminationTermCfg(
        func=terminations.illegal_contact,
        params={ 
            "threshold": 200.0, 
            "sensor_cfg": SceneEntityCfg(name="contact_forces_right") 
        } 
    )
    
    
@configclass
class LeggedRobotV2EnvCfgBalance(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""
    
    # Scene settings
    scene: LeggedRobotV2SceneConfig = LeggedRobotV2SceneConfig(
        num_envs=1,
        env_spacing=2.0,
    )

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.sim.device = "gpu"
        self.sim.use_fabric = True
        
        self.decimation = 1  # Control freq = 60/1 = 60 Hz
        self.episode_length_s = 20  # Episode duration
        
        # Viewer settings
        self.viewer.eye = (5.0, 0.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point
                
        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps
        