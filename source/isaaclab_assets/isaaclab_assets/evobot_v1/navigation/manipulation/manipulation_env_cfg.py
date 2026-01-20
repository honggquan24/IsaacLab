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
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import (
    ContactSensorCfg,
    ImuCfg,
)
from ...evobot_v1_cfg import EVOBOT_V1_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp import actions, observations, events, rewards, terminations, commands
from ...mdp import *

from ...mdp import (
    joint_pos_target_l2,
    velocity_heading_alignment,
    reward_man,
    gripper_height_tracking_l2,
    joint_angle_command_l2,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_v

@configclass
class EvobotV1SceneConfig(InteractiveSceneCfg):
    """Scene configuration for the Evobot V1 environment."""
    num_envs: int = 1

    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    
    # ground 
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

    # # Contact sensor - mounted on head_link to detect illegal contacts
    # contact_forces_arm_link = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/evobot/evobot/head_link",
    #     update_period=0.01,
    # )
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/evobot/evobot/.*")


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
        scale=100.0,  # High torque for moving the robot
    )

    # Arm - Medium torque for manipulation
    arm_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'arm_joint',              # Revolute - Arm rotation
        ],
        scale=20.0,  # Medium torque for arm movement
    )

    # Grabbers - Low force for grasping (added to match balance checkpoint)
    grabber_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            'left_grabbing_joint',    # Prismatic - Left gripper
            'right_grabbing_joint',   # Prismatic - Right gripper
        ],
        scale=1.0,  # Low force to avoid damaging objects
    )

@configclass
class CommandsCfg:
    """Velocity commands với consideration cho balance."""
    
    base_velocity = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),  # Giữ command 8-12s (đủ để học)
        
        # QUAN TRỌNG: Tùy chỉnh cho balance
        rel_standing_envs=0.3,     # 30% thời gian đứng yên (tập balance tại chỗ)
        
        heading_command=False,     # Dùng angular velocity (not heading)
        debug_vis=True,
        
        # RANGE AN TOÀN CHO BALANCE
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),      # TỐC ĐỘ CHẬM để giữ balance
            lin_vel_y=(0.0, 0.0),       # Không di chuyển ngang
            ang_vel_z=(-1.0, 1.0),      # Xoay chậm
            heading=(0.0, 0.0),
        ),
    )
    
    # Arm joint angle command (use yaw component as joint angle target)
    arm_ee_pose = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used
            pos_y=(0.0, 0.0),  # Not used
            pos_z=(0.0, 0.0),  # Not used
            roll=(0.0, 0.0),   # Not used
            pitch=(0.0, 0.0),  # Not used
            yaw=(-math.pi, math.pi),  # Use yaw as joint angle target
        ),
    )

    # Gripper height commands (relative z-position)
    grip_ee_pose_left = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used for height tracking
            pos_y=(0.0, 0.0),  # Not used for height tracking
            pos_z=(-0.15, 0.15),  # Target height range: ±15cm relative to base
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    grip_ee_pose_right = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_01",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used for height tracking
            pos_y=(0.0, 0.0),  # Not used for height tracking
            pos_z=(-0.15, 0.15),  # Target height range: ±15cm relative to base
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation configuration for the policy."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group."""
        
        # observation terms (order preserved)
        base_lin_vel = ObservationTermCfg(func=mdp_v.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp_v.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp_v.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # IMU sensors
        imu_lin_acc = ObservationTermCfg(func=observations.imu_lin_acc)
        # imu_ang_vel = ObservationTermCfg(func=observations.imu_ang_vel)
        # imu_orientation = ObservationTermCfg(func=observations.imu_orientation)
                
        # Joint states
        joint_pos = ObservationTermCfg(func=observations.joint_pos)
        joint_vel = ObservationTermCfg(func=observations.joint_vel)
        joint_effort = ObservationTermCfg(func=observations.joint_effort)

        # Previous actions (for smoothness)
        last_action = ObservationTermCfg(func=observations.last_action)
            
        # Commands
        base_velocity_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
        )
        
        arm_ee_pose_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "arm_ee_pose"},
        )
        
        grip_ee_pose_left_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "grip_ee_pose_left"},
        )
        
        grip_ee_pose_right_cmd = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "grip_ee_pose_right"},
        )


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        # observation terms (order preserved)
        base_lin_vel = ObservationTermCfg(func=mdp_v.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp_v.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp_v.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Pose
        body_pose_w = ObservationTermCfg(func=observations.body_pose_w)

        # Root state (privileged information)
        root_pos_w = ObservationTermCfg(func=observations.root_pos_w)
        root_quat_w = ObservationTermCfg(func=observations.root_quat_w)
        root_lin_vel_w = ObservationTermCfg(func=observations.root_lin_vel_w)
        root_ang_vel_w = ObservationTermCfg(func=observations.root_ang_vel_w)

        # # # IMU data (keep only essential)
        # imu_orientation = ObservationTermCfg(func=observations.imu_orientation)
        # imu_projected_gravity = ObservationTermCfg(func=observations.imu_projected_gravity)

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
        func=rpy_alignment_imu,
        weight=10.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
            "tolerance": 0.05,
        },
    )
    
    lin_vel_z_l2 = RewardTermCfg(
        func=rewards.lin_vel_z_l2, 
        weight=-2.0
    )
    
    ang_vel_xy = RewardTermCfg(
        func=rewards.ang_vel_xy_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    dof_acc_l2 = RewardTermCfg(
        func=joint_acc_l2,
        weight=-2.5e-8
    )
    
    # (3) Command task
    lin_vel_tracking = RewardTermCfg(
        func=rewards.track_lin_vel_xy_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": 0.05,
        },
    )

    ang_vel_tracking = RewardTermCfg(
        func=rewards.track_ang_vel_z_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.05,
        },
    )

    # Smooth
    action_rate = RewardTermCfg(
        func=action_rate_l2,
        weight=-0.05,
    )
    
    undesired_contacts = RewardTermCfg(
        func=undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="head_link"),
            "threshold": 1.0},
    )

    
    # Manipulation - Joint angle tracking for arm
    arm_ee_tracking = RewardTermCfg(
        func=joint_angle_command_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_joint"),
            "command_name": "arm_ee_pose",
        },
    )

    # Gripper height tracking
    grip_ee_tracking_left = RewardTermCfg(
        func=gripper_height_tracking_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper"),
            "command_name": "grip_ee_pose_left",
        },
    )

    grip_ee_tracking_right = RewardTermCfg(
        func=gripper_height_tracking_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_01"),
            "command_name": "grip_ee_pose_right",
        },
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
    arm_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 10.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="head_link") 
        }
    ) 
    
    left_grip_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 10.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper") 
        }
    ) 
    
    right_grip_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 10.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper_01") 
        }
    ) 

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurriculumTermCfg(func=mdp_v.terrain_levels_vel)

@configclass
class EvobotV1LocomotionManipulationBalanceEnvCfg(ManagerBasedRLEnvCfg):
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
    # curriculum: CurriculumCfg = CurriculumCfg()
    
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
