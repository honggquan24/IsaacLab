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
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from .legged_v1_cfg import LEGGED_ROBOT_V1_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
import isaaclab.utils.math as math_utils
from icecream import ic
from . import mdp


@configclass
class LeggedRobotV1SceneConfig(InteractiveSceneCfg):
    """Scene configuration for the legged robot environment."""
    
    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Add terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=3,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="average",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=True,          
    # )

    cfg_ground = AssetBaseCfg( 
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg(), 
    )

    # Add robot 
    robot: Articulation = LEGGED_ROBOT_V1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Add IMU sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot_2_leg_base/robot_2_leg/robot_2_leg/Group_1",
        update_period=0.1,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=True,
    )


@configclass
class ActionCfg:
    """Action configuration for joint effort control."""
    
    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "Revolute_1",
            "Revolute_2",
            "Revolute_3",
            "Revolute_4",
            "Revolute_5",
            "Revolute_6",
        ],
        scale={".*": 250.0},
        clip={
            ".*": (-250.0, 250.0)  # Giới hạn mô-men (Nm) cho tất cả các khớp
        },
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for the policy."""
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group."""
        
        # Joint positions relative to default
        joint_pos_rel = ObservationTermCfg(func=observations.joint_pos_rel)
        
        # Joint velocities relative to default (commented out for now)
        # joint_vel_rel = ObservationTermCfg(func=observations.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for environment resets."""
    
    # Reset joint positions and velocities with small random offsets
    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (-0.02, 0.02),  # Giảm từ 0.05 để ổn định hơn
            "velocity_range": (-0.05, 0.05),  # Giảm từ 0.1 để ổn định hơn
        },
    )

    # Reset root state (position, orientation, velocities)
    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x": (-2.0, 2.0),      # Giảm từ (-5, 5) để tránh spawn ngoài terrain
                "y": (-2.0, 2.0),      # Giảm từ (-5, 5) để tránh spawn ngoài terrain
                "z": (0.5, 0.51),       # Phù hợp với init_state.pos[2]=0.75
                "roll": (-0.05, 0.05), # Thêm variation nhỏ cho roll
                "pitch": (-0.05, 0.05),# Thêm variation nhỏ cho pitch
                "yaw": (-0.3, 0.3),    # Giảm từ (-0.5, 0.5) để ổn định hơn
            },
            "velocity_range": {
                "linear": (0.0, 0.0),  # Không có vận tốc tịnh tiến ban đầu
                "angular": (0.0, 0.0), # Không có vận tốc góc ban đầu
            },
        },
    )


@configclass
class RewardCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward - khuyến khích robot sống sót
    alive = RewardTermCfg(
        func=rewards.is_alive, 
        weight=1.0
    )

    # (2) Failure penalty - phạt khi bị terminate
    terminating = RewardTermCfg(
        func=rewards.is_terminated, 
        weight=-1.0
    )

    # (3) Keep upright - khuyến khích đứng thẳng
    upright_balance = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=3.0,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
        },
    )

    # (4) Stay still - giảm rung lắc
    stillness = RewardTermCfg(
        func=mdp.rewards.imu_stillness_reward,
        weight=0.5,
        params={
            "imu_cfg": SceneEntityCfg(name="imu")
        },
    )


@configclass
class TerminationsCfg:
    """
    Termination configuration cho legged robot environment.
    Các termination sẽ trigger reset environment khi điều kiện được thỏa mãn.
    """

    # TIME OUT
    # Kết thúc episode khi đã chạy đủ thời gian episode_length_s
    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,  # Đánh dấu đây là timeout (không phải failure)
    )
    # ROOT HEIGHT BELOW MINIMUM
    # Kết thúc khi robot rơi xuống quá thấp (base của robot < minimum_height)
    base_height = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.05,  # Terminate nếu base_z < 0.3 m
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    # Lưu ý: Chỉ hoạt động với flat terrain hoặc terrain trong world frame

    # BAD ORIENTATION
    # Kết thúc khi robot nghiêng quá nhiều so với trọng lực
    # Tính toán dựa trên góc giữa gravity vector được chiếu và trục z
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 3,  # Góc giới hạn (radians) = ? độ
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # JOINT POSITION OUT OF SOFT LIMITS
    # Kết thúc khi bất kỳ khớp nào vượt quá soft joint limits
    # Soft limits thường nhỏ hơn hard limits một chút để bảo vệ khớp
    # Khi xảy ra: joint_pos < soft_lower_limit HOẶC joint_pos > soft_upper_limit
    # Ví dụ: Revolute_1 có limit [-1.57, 1.57] rad
    #        Nếu joint_pos = 1.60 rad -> RESET
    # Soft limits được định nghĩa trong USD file hoặc ArticulationCfg
    # joint_pos_limit = TerminationTermCfg(
    #     func=terminations.joint_pos_out_of_limit,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    # JOINT VELOCITY OUT OF LIMITS
    # Kết thúc khi vận tốc khớp quá lớn (có thể gây hỏng robot hoặc mất ổn định)
    joint_vel_limit = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 500.0,  # rad/s - Giới hạn vận tốc góc tối đa
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # JOINT EFFORT OUT OF LIMITS
    # Kết thúc khi mô-men khớp vượt quá soft effort limits
    joint_effort_limit = TerminationTermCfg(
        func=terminations.joint_effort_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

@configclass
class LeggedRobotV1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""
    
    # Render settings
    render_cfg = sim_utils.RenderCfg(
        rendering_mode="performance",
    )
    
    # Scene settings
    scene: LeggedRobotV1SceneConfig = LeggedRobotV1SceneConfig(
        env_spacing=1.0,
    )

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2  # Control frequency = sim_freq / decimation = 60/2 = 30 Hz
        self.episode_length_s = 30  # Episode duration in seconds
        
        # Viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)  # Camera position
        
        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps