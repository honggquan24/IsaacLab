"""Navigation environment configuration for Evobot V1.

This extends the balance task with command-following capabilities,
allowing the robot to navigate to random target positions while
maintaining balance.
"""

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
    ImuCfg,
)

from ..robot.evobot_v1_cfg import EVOBOT_V1_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations, commands
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
    """

    # Wheels - High torque for locomotion
    wheel_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'left_wheel_joint',       # Revolute - Left wheel
            'right_wheel_joint',      # Revolute - Right wheel
        ],
        scale=200.0,  # High torque for moving the robot
    )

    # Arm - Medium torque for manipulation
    arm_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'arm_joint',              # Revolute - Arm rotation
        ],
        scale=50.0,  # Medium torque for arm movement
    )

    # Grabbers - Low force for grasping (added to match balance checkpoint)
    grabber_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'left_grabbing_joint',    # Prismatic - Left gripper
            'right_grabbing_joint',   # Prismatic - Right gripper
        ],
        scale=10.0,  # Low force to avoid damaging objects
    )

@configclass
class CommandsCfg:
    """Velocity commands với consideration cho balance."""
    
    base_velocity = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0, 2.0),  # Giữ command 8-12s (đủ để học)
        
        # QUAN TRỌNG: Tùy chỉnh cho balance
        rel_standing_envs=0.3,     # 30% thời gian đứng yên (tập balance tại chỗ)
        
        heading_command=False,     # Dùng angular velocity (not heading)
        debug_vis=True,
        
        # RANGE AN TOÀN CHO BALANCE
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),      # TỐC ĐỘ CHẬM để giữ balance
            lin_vel_y=(0.0, 0.0),       # Không di chuyển ngang
            ang_vel_z=(-math.pi, -math.pi),      # Xoay chậm
            heading=(0.0, 0.0),
        ),
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

        # Commands
        base_velocity_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
        )
                
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
    """REWARD DESIGN: Balance + Velocity Tracking."""
    
    # (1) Survival
    alive = RewardTermCfg(
    func=rewards.is_alive,
    weight=2.0,
    )

    terminating = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-500.0,
    )
    
    # (2) Balance
    rpy_alignment = RewardTermCfg(
    func=mdp.rewards.rpy_alignment_imu,
    weight=10.0,
    params={
        "target_rpy": (0.0, 0.0, 0.0),
        "imu_cfg": SceneEntityCfg(name="imu"),
        "tolerance": 0.05,
    },)
    
    ang_vel_xy = RewardTermCfg(
        func=ang_vel_xy_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # (3) Command 
    lin_vel_tracking = RewardTermCfg(
        func=rewards.track_lin_vel_xy_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    ang_vel_tracking = RewardTermCfg(
        func=rewards.track_ang_vel_z_exp,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # Smooth
    action_rate = RewardTermCfg(
        func=action_rate_l2,
        weight=-0.5,
    )

    
@configclass
class TerminationsCfg:
    """Terminations: Strict cho balance, lenient cho velocity."""
    
    # 1. TIME OUT (normal)
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )
    
    # 2. FALL DOWN (QUAN TRỌNG)
    # Kết hợp nhiều điều kiện fall
    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.25,  # 8cm (thấp hơn chút để cho recovery chance)
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # 3. BAD ORIENTATION (khi nghiêng quá nhiều)
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 1.2,  # ~72° (strict hơn)
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
class EvobotV1VelocityBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration cho velocity control với balance constraints."""
    
    # Scene
    scene: EvobotV1SceneConfig = EvobotV1SceneConfig(
        num_envs=1,
        env_spacing=2.0,
    )
    
    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        # General
        self.sim.device = "gpu"
        self.sim.use_fabric = True
        
        # KEY: Episode length đủ dài để học recovery
        self.decimation = 1  # Control at 15Hz (60/4)
        self.episode_length_s = 10.0  # 25 seconds
        
        # Physics
        self.sim.dt = 1 / 60.0
        
        # Viewer
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
