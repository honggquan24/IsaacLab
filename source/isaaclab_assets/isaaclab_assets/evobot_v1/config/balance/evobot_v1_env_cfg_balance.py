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

from ..robot.evobot_v1_cfg import EVOBOT_V1_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
from ... import mdp
from isaaclab.envs.mdp import *



@configclass
class EvobotV1SceneConfig(InteractiveSceneCfg):
    """Scene configuration for the Evobot V1 environment."""
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
    # )

    cfg_ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Add robot
    robot: Articulation = EVOBOT_V1_CFG.replace( # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Add IMU sensor - mounted on top_link (upper body)
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/evobot/evobot/top_link",
        update_period=0.02,  # 50Hz to match control frequency
        gravity_bias=(0.0, 0.0, 0.0),
    )

    # Contact sensor - mounted on head_link to detect illegal contacts
    contact_forces_arm_link = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/evobot/evobot/head_link",
        update_period=0.01,
    )


@configclass
class ActionCfg:
    """Action configuration for joint effort control.

    Available joints in USD (5 DOF total):
    - left_wheel_joint (Revolute)
    - right_wheel_joint (Revolute)
    - arm_joint (Revolute)
    - left_grabbing_joint (Prismatic)
    - right_grabbing_joint (Prismatic)

    All joints are at root level: /Robot/evobot/evobot/<joint_name>

    Scale values are configured separately for each joint group:
    - Wheels: High torque for locomotion (500.0)
    - Arm: Medium torque for manipulation (300.0)
    - Grabbers: Low force for grasping (50.0)
    """

    # Wheels - High torque for locomotion
    wheel_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'left_wheel_joint',       # Revolute - Left wheel
            'right_wheel_joint',      # Revolute - Right wheel
        ],
        scale=300.0,  # High torque for moving the robot
    )

    # Arm - Medium torque for manipulation
    arm_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'arm_joint',              # Revolute - Arm rotation
        ],
        scale=100.0,  # Medium torque for arm movement
    )

    # Grabbers - Low force for grasping
    grabber_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'left_grabbing_joint',    # Prismatic - Left gripper
            'right_grabbing_joint',   # Prismatic - Right gripper
        ],
        scale=50.0,  # Low force to avoid damaging objects
    )


@configclass
class ObservationsCfg:
    """Observation configuration for the policy."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group."""

        # # IMU sensors
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

        # Previous actions (for smoothness)
        last_action = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """FIX MEMORY LEAK: Giảm critic observations từ 16 → 8 terms

        Removed redundant observations:
        - body_pose_w: duplicate of root_pos_w + root_quat_w
        - base_lin_vel: duplicate of root_lin_vel_w
        - imu_lin_acc: có thể suy ra từ root_lin_vel_w
        - current_time/remaining_time: không cần cho balance task ngắn (5s)
        """

        # Root state (privileged information)
        root_pos_w = ObservationTermCfg(func=observations.root_pos_w)
        root_quat_w = ObservationTermCfg(func=observations.root_quat_w)
        root_lin_vel_w = ObservationTermCfg(func=observations.root_lin_vel_w)
        root_ang_vel_w = ObservationTermCfg(func=observations.root_ang_vel_w)

        # # IMU data (keep only essential)
        imu_orientation = ObservationTermCfg(func=observations.imu_orientation)
        imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)

        # Joint states
        joint_pos = ObservationTermCfg(func=observations.joint_pos)
        joint_vel = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)
        
        # Previous actions
        last_action = ObservationTermCfg(func=observations.last_action)

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
                "z": (0.12, 0.12),
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
    """Simplified reward terms for balance task (7 core terms)."""

    # (1) Survival reward
    alive = RewardTermCfg(
        func=rewards.is_alive,
        weight=2.0,
    )

    # (2) Termination penalty
    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-100.0,
    )

    # (3) RPY alignment - Main balance reward (target: upright vertical)
    rpy_alignment = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=10.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),  # Upright vertical orientation
            "imu_cfg": SceneEntityCfg(name="imu"),
            "tolerance": 0.1,  # 0.1 radians tolerance
        },
    )

    # (4) Action smoothness penalty
    action_rate = RewardTermCfg(
        func=action_rate_l2,
        weight=-1.0,
    )

    # (5) Joint velocity penalty (combined, not per-joint)
    joint_vel = RewardTermCfg(
        func=joint_vel_l2,
        weight=-2e-4,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (6) Angular velocity stability (roll/pitch rates)
    ang_vel_xy = RewardTermCfg(
        func=ang_vel_xy_l2,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (7) Joint velocity penalty (combined, not per-joint)
    joint_acc_l2 = RewardTermCfg(
        func=joint_acc_l2,
        weight=-1e-8,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (8) Angular velocity stability (roll/pitch rates)
    body_lin_acc_l2 = RewardTermCfg(
        func=body_lin_acc_l2,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination configuration for Evobot V1 environment."""

    # TIME OUT - Episode ends after episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,  # Mark as timeout (not failure)
    )

    # ROOT HEIGHT BELOW MINIMUM
    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.35,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # BAD ORIENTATION
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 3,  # 60° tolerance
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # Contact illegal 
    contact_arm = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 30.0, 
            "sensor_cfg": SceneEntityCfg(name="contact_forces_arm_link") 
        }
    ) 


@configclass
class EvobotV1EnvCfgBalance(ManagerBasedRLEnvCfg):
    """Configuration for the Evobot V1 balance environment."""

    # Scene settings
    scene: EvobotV1SceneConfig = EvobotV1SceneConfig(
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

        # FIX MEMORY LEAK: Reduce episode length to match buffer size better
        # Old: 5 sec × 60 Hz = 300 steps >> 48 buffer → 6.25 buffer fills per episode
        # New: 2 sec × 60 Hz = 120 steps → 2.5 buffer fills per episode
        self.decimation = 1  # Control freq = 60/1 = 60 Hz
        self.episode_length_s = 10  # Episode duration (reduced from 5s)

        # Viewer settings
        self.viewer.eye = (5.0, 0.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point

        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps
