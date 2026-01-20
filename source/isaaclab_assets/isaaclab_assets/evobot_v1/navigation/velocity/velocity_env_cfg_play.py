"""Velocity balance environment configuration for Evobot V1.

This extends the balance task with command-following capabilities,
allowing the robot to navigate to random target positions while
maintaining balance.
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.terrains import TerrainImporterCfg
from ...evobot_v1_cfg import EVOBOT_V1_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp import actions, observations, events, rewards, terminations, commands
from isaaclab.envs.mdp import *
from ...mdp import (
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

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.2, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.2)),
    )
    
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
    velocity_command = commands.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),  # Giữ command 3-5s

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
    
        
    # Arm joint angle command (use yaw component as joint angle target)
    arm_ee_pose = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link",
        resampling_time_range=(3.0, 5.0),  # Change target every 3-5 seconds
        # debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used
            pos_y=(0.0, 0.0),  # Not used
            pos_z=(0.0, 0.0),  # Not used
            roll=(0.0, 0.0),   # Not used
            pitch=(0.0, 0.0),  # Not used
            yaw=(-math.pi, math.pi), # Use yaw as joint angle target (±90 degrees)
        ),
    )
    
    grip_ee_pose_left = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper",   # đổi đúng link EE của evobot
        resampling_time_range=(3.0, 5.0),
        # debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used for height tracking
            pos_y=(0.0, 0.0),  # Not used for height tracking
            pos_z=(0, 0.1),  # Target height range: ±15cm relative to base
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    grip_ee_pose_right = commands.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_01",   # đổi đúng link EE của evobot
        resampling_time_range=(3.0, 5.0),
        # debug_vis=True,
        ranges=commands.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),  # Not used for height tracking
            pos_y=(0.0, 0.0),  # Not used for height tracking
            pos_z=(0, 0.1),  # Target height range: ±15cm relative to base
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
        velocity_command = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
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
        velocity_command = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "velocity_command"},
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
    """Event configuration for environment resets - CLEAN VERSION for testing (no randomization)."""

    # Reset joints to default positions (no random offsets)
    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (0.0, 0.0),  # No randomization
            "velocity_range": (0.0, 0.0),  # No randomization
        },
    )

    # Reset base to default pose (no noise)
    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x": (0.0, 0.0),     # No randomization
                "y": (0.0, 0.0),     # No randomization
                "z": (0.12, 0.12),   # Fixed height
                "roll": (0.0, 0.0),  # No randomization
                "pitch": (0.0, 0.0), # No randomization
                "yaw": (0.0, 0.0),   # No randomization
            },
            "velocity_range": {
                "linear": (0.0, 0.0),  # No randomization
                "angular": (0.0, 0.0), # No randomization
            },
        },
    )

    # REMOVED: randomize_com - No COM randomization for clean testing
    # REMOVED: external_push_arm - No external disturbances for clean testing



    
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
    
    # (3) Command tracking
    lin_vel_tracking = RewardTermCfg(
        func=rewards.track_lin_vel_xy_exp,
        weight=-20.0,  
        params={
            "command_name": "velocity_command",
            "std": 0.5,
        },
    )

    ang_vel_tracking = RewardTermCfg(
        func=rewards.track_ang_vel_z_exp,
        weight=-20.0,  
        params={
            "command_name": "velocity_command",
            "std": 0.5,
        },
    )

    # Joint angle tracking - Arm tracks commanded angle (from yaw component)
    arm_joint_tracking = RewardTermCfg(
        func=joint_angle_command_l2,
        weight=-50.0,  # Penalty for joint angle error (squared error)
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_joint"),
            "command_name": "arm_ee_pose",  # Extract yaw from this command
        },
    )

    # Smooth
    action_rate = RewardTermCfg(
        func=rewards.action_rate_l2,
        weight=-0.005,
    )

    
@configclass
class TerminationsCfg:
     
    # """Terminations: Strict cho balance, lenient cho velocity."""
    # 1. TIME OUT (normal)
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )
    
    # 2. FALL DOWN 
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
            "threshold": 5000.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="gripper.*") 
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
        
        self.decimation = 1  
        self.episode_length_s = 60.0  
        # Physics
        self.sim.dt = 1 / 60.0
        
        # Viewer
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
