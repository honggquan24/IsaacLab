"""Velocity balance environment configuration for Evobot V1.

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
    CurriculumTermCfg
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import (
    ContactSensorCfg,
    ImuCfg,
)
from isaaclab.terrains import TerrainImporterCfg
from ...evobot_v1_cfg import EVOBOT_V1_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp import actions, observations, events, rewards, terminations, commands
from isaaclab.envs.mdp import *
from ...mdp import (
    rpy_alignment_imu,
    reset_when_fall,
    reward_wheel_speed,
    joint_pos_target_l2,
    velocity_heading_alignment,
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

    # # Contact sensor - mounted on arm_link to detect illegal contacts
    # contact_forces_arm_link = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/evobot/evobot/arm_link",
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
    - left_gripper_joint (Prismatic)
    - right_gripper_joint (Prismatic)
    """
    

    # # Wheels - High torque for locomotion
    # wheel_effort = actions.JointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         'left_wheel_joint',     
    #         'right_wheel_joint',    
    #     ],
    #     scale=300.0,  
    # )

    # # Arm - Medium torque for manipulation
    # arm_effort = actions.JointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         'arm_joint',       
    #     ],
    #     scale=300.0,  
    # )

    # # Grabbers - Low force for grasping (added to match balance checkpoint)
    # grabber_effort = actions.JointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         'left_gripper_joint', 
    #         'right_gripper_joint',
    #     ],
    #     scale=100.0,  # High torque for moving the robot
    # )
    all_joints = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "left_wheel_joint",
            "right_wheel_joint",
            "arm_joint",
            "left_gripper_joint",
            "right_gripper_joint",
        ],
        scale={
            "left_wheel_joint": 400.0,    # Match effort_limit
            "right_wheel_joint": 400.0,
            "arm_joint": 200.0,
            "left_gripper_joint": 80.0,
            "right_gripper_joint": 80.0,
        },
    )


@configclass
class CommandsCfg:
    """Velocity commands với consideration cho balance."""
    base_velocity = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),  # Giữ command 3-5s

        # QUAN TRỌNG: Tùy chỉnh cho balance
        rel_standing_envs=0.1,     # 10% thời gian đứng yên (tập balance tại chỗ)

        heading_command=False,     # FALSE = Dùng angular velocity (not heading angle)
        debug_vis=True,

        # RANGE AN TOÀN CHO BALANCE + Xoay
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),      # TỐC ĐỘ CHẬM để giữ balance
            lin_vel_y=(0.0, 0.0),        # BỎ y-velocity (differential drive không đi ngang)
            ang_vel_z=(-1.0, 1.0),       # Angular velocity range
            heading=(0.0, 0.0),          # Ignored khi heading_command=False
        ),
    )
    
        
    arm_ee_pose = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link",   # đổi đúng link EE của evobot
        resampling_time_range=(1.0, 3.0),
        # debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )
    
    grip_ee_pose_left = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper",   # đổi đúng link EE của evobot
        resampling_time_range=(1.0, 3.0),
        # debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.1),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0),
        ),
    )
    
    grip_ee_pose_right = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_01",   # đổi đúng link EE của evobot
        resampling_time_range=(1.0, 3.0),
        # debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.1),
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
        base_lin_vel = ObservationTermCfg(func=mdp_v.base_lin_vel) # noise=Unoise(n_min=-0.1, n_max=0.1)
        # base_ang_vel = ObservationTermCfg(func=mdp_v.base_ang_vel) # noise=Unoise(n_min=-0.1, n_max=0.1)
        # projected_gravity = ObservationTermCfg(
        #     func=mdp_v.projected_gravity,
        # )
        
        # IMU sensors
        imu_lin_acc = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation = ObservationTermCfg(func=observations.imu_orientation)
                
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
        base_lin_vel = ObservationTermCfg(func=mdp_v.base_lin_vel) # noise=Unoise(n_min=-0.1, n_max=0.1)
        # base_ang_vel = ObservationTermCfg(func=mdp_v.base_ang_vel) # noise=Unoise(n_min=-0.1, n_max=0.1)
        # projected_gravity = ObservationTermCfg(
        #     func=mdp_v.projected_gravity,
        # )
        
        # IMU sensors
        imu_lin_acc = ObservationTermCfg(func=observations.imu_lin_acc)
        imu_ang_vel = ObservationTermCfg(func=observations.imu_ang_vel)
        imu_orientation = ObservationTermCfg(func=observations.imu_orientation)
                
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
        weight=-100.0,
    )
    
    # wheel_speed = RewardTermCfg(
    #     func=reward_wheel_speed,
    #     weight=5.0,
    # )
    
    # (2) Balance
    # rpy_alignment = RewardTermCfg(
    #     func=rpy_alignment_imu,
    #     weight=20,    #2.0,
    #     params={
    #         "target_rpy": (0.0, 0.0, 0.0),
    #         "imu_cfg": SceneEntityCfg(name="imu"),
    #         "tolerance": 0.05,
    #     },
    # )
    
    # lin_vel_z_l2 = RewardTermCfg(
    #     func=rewards.lin_vel_z_l2, 
    #     weight=-20,    #-1.1,
    # )
    
    # ang_vel_xy = RewardTermCfg(
    #     func=ang_vel_xy_l2,
    #     weight=0.0001,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    
    # dof_acc_l2 = RewardTermCfg(
    #     func=joint_acc_l2,
    #     weight=-2.5e-8
    # ) 
    
    # (3) Command tracking
    lin_vel_tracking = RewardTermCfg(
        func=rewards.track_lin_vel_xy_exp,
        weight=35.0,  # GIẢM từ 20 → 15 để cân bằng với angular
        params={
            "command_name": "base_velocity",
            "std": 0.05,
        },
    )

    ang_vel_tracking = RewardTermCfg(
        func=rewards.track_ang_vel_z_exp,
        weight=20.0,  # TĂNG từ 15 → 20 để khuyến khích xoay
        params={
            "command_name": "base_velocity",
            "std": 0.05,
        },
    )

    # Heading alignment: Khuyến khích robot xoay đúng hướng trước khi đi
    heading_alignment = RewardTermCfg(
        func=velocity_heading_alignment,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,  # Càng nhỏ = yêu cầu alignment càng chặt chẽ
        },
    )

    # Smooth
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.001,
    )
    
    undesired_contacts = RewardTermCfg(
        func=rewards.undesired_contacts,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="arm_link|gripper.*"),
            "threshold": 10.0},
    )
    
    arm_error = RewardTermCfg(
        func=joint_pos_target_l2,
        weight=-5.0,    
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="arm_joint"),
            "target": 0.0,
        },
    )

    # # Manipualtion
    # arm_ee_tracking = RewardTermCfg(
    #     func=manipulation_mdp.rpy_command_error,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="arm_link"),
    #         "command_name": "arm_ee_pose",
    #     },
    # )
    
    # grip_ee_tracking_left = RewardTermCfg(
    #     func=manipulation_mdp.position_command_error_man,
    #     weight=0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="gripper"),
    #         "command_name": "grip_ee_pose_left",
    #     },
    # )
    
    # grip_ee_tracking_right = RewardTermCfg(
    #     func=manipulation_mdp.position_command_error_man,
    #     weight=0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="gripper_01"),
    #         "command_name": "grip_ee_pose_right",
    #     },
    # )
    
@configclass
class TerminationsCfg:
    """Terminations: Strict cho balance, lenient cho velocity."""
    # 1. TIME OUT (normal)
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )
    
    # # 2. FALL DOWN 
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
            "limit_angle": math.pi / 1.2,  
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    # Contact illegal 
    left_grip_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 200.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper") 
        }
    ) 
    
    right_grip_contact = TerminationTermCfg( 
        func=terminations.illegal_contact, 
        params={ 
            "threshold": 200.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper_01") 
        }
    ) 
    
    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 1000.0,  # rad/s - Giới hạn vận tốc góc tối đa
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    
# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurriculumTermCfg(func=mdp_v.terrain_levels_vel)

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
