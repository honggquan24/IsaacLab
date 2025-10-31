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

from .legged_v2_cfg import LEGGED_ROBOT_V2_CFG

from isaaclab.envs.mdp import actions, observations, events, rewards, terminations
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from icecream import ic
from . import mdp
import math

@configclass
class LeggedRobotV2SceneConfig(InteractiveSceneCfg):
    """Scene configuration for the legged robot environment."""
    
    # Add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=0.8,
            dynamic_friction=0.05,
            restitution=0.0,
        ),
        debug_vis=True,          
    )

    # cfg_ground = AssetBaseCfg( 
    #     prim_path="/World/ground", 
    #     spawn=sim_utils.GroundPlaneCfg(), 
    # )

    # Add robot 
    robot: Articulation = LEGGED_ROBOT_V2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Add IMU sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v2/robot_legged_v2/base_42",
        update_period=0.1,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=True,
    )

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/robot_legged_v2/robot_legged_v2/base_42",
    #     update_period=0.1,
    #     offset=RayCasterCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 20.0)
    #     ),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(
    #         resolution=0.1,
    #         size=[1.6, 1.0],
    #     ),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )


@configclass
class ActionCfg:
    """Action configuration for joint effort control."""
    
    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "Revolute_1",  # hip_left
            "Revolute_2",  # hip_right
            "Revolute_3",  # knee_left
            "Revolute_4",  # knee_right
            "Revolute_5",  # ankle_left
            "Revolute_6",  # ankle_right
            "Revolute_7",  # wheel_left
            "Revolute_8",  # wheel_right
        ],
        scale={
            "Revolute_[1-6]": 1000.0,  # Leg joints: scale lớn hơn
            "Revolute_[7-8]": 5000.0,  # Wheels: scale GẤP 10 LẦN
        },
        clip={
            "Revolute_[1-6]": (-1.0, 1.0),  # Clip INPUT action, không phải output
            "Revolute_[7-8]": (-1.0, 1.0),
        },
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for the policy."""
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group."""
        
        # Base orientation (projected gravity)
        base_orientation = ObservationTermCfg(
            func=observations.base_lin_vel  # Hoặc dùng IMU data
        )
        
        # Joint positions
        joint_pos_rel = ObservationTermCfg(func=observations.joint_pos_rel)
        
        # Joint velocities - QUAN TRỌNG!
        joint_vel_rel = ObservationTermCfg(func=observations.joint_vel_rel)
        
        # Previous actions (for smoothness)
        last_action = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for environment resets."""
    
    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (-0.1, 0.1),  # Thêm randomization
            "velocity_range": (-0.5, 0.5),  # Thêm randomization
        },
    )

    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x": (-5, 5),    # Thêm variation
                "y": (-5, 5),    # Thêm variation
                "z": (0.3, 0.6),
                "roll": (-0.1, 0.1),   # Giảm xuống
                "pitch": (-0.1, 0.1),  # Giảm xuống
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "linear": (0.0, 0.0),
                "angular": (0.0, 0.0),
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
        weight=-2.0
    )
    
    # # (3) Keep upright - khuyến khích đứng thẳng (chỉ roll & pitch)
    # upright_posture = RewardTermCfg(
    #     func=mdp.rewards.upright_posture_reward,
    #     weight=2.5,
    #     params={
    #         "imu_cfg": SceneEntityCfg(name="imu"),
    #         "tolerance": 0.02,
    #     },
    # )
    
    # (4) Full RPY alignment - căn chỉnh toàn bộ hướng (roll, pitch, yaw)
    rpy_alignment = RewardTermCfg(
        func=mdp.rewards.rpy_alignment_imu,
        weight=5,
        params={
            "target_rpy": (0.0, 0.0, 0.0),
            "imu_cfg": SceneEntityCfg(name="imu"),
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
    # base_height = TerminationTermCfg(
    #     func=terminations.root_height_below_minimum,
    #     params={
    #         "minimum_height": 0.01,  # Terminate nếu base_z < 0.3 m
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    # Lưu ý: Chỉ hoạt động với flat terrain hoặc terrain trong world frame

    # BAD ORIENTATION
    # Kết thúc khi robot nghiêng quá nhiều so với trọng lực
    # Tính toán dựa trên góc giữa gravity vector được chiếu và trục z
    # bad_orientation = TerminationTermCfg(
    #     func=terminations.bad_orientation,
    #     params={
    #         "limit_angle": math.pi / 5,  # Góc giới hạn (radians) = ? độ
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

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
            "max_velocity": 200.0,  # rad/s - Giới hạn vận tốc góc tối đa
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # JOINT EFFORT OUT OF LIMITS
    # Kết thúc khi mô-men khớp vượt quá soft effort limits
    # joint_effort_limit = TerminationTermCfg(
    #     func=terminations.joint_effort_out_of_limit,
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    
    

@configclass
class LeggedRobotV2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged robot environment."""
    # Scene settings
    scene: LeggedRobotV2SceneConfig = LeggedRobotV2SceneConfig(
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
        self.episode_length_s = 10  # Episode duration in seconds
        
        # Viewer settings
        self.viewer.eye = (0.0, 5.0, 50.0)  # Camera position
                
        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps 
        
        self.sim.substeps = 4  # Số substeps mỗi physics step
        # Với dt=1/60 và substeps=2:
        #   → Mỗi physics step = 16.67ms
        #   → Được chia thành 2 substeps = 8.33ms mỗi substep
        #   → Effective physics rate = 60 * 2 = 120 Hz
        
        # OPTION 2: Tăng PhysX solver iterations (nếu cần stability cao hơn)
        self.sim.render_cfg = sim_utils.RenderCfg(
            rendering_mode="performance",
            # user friendly setting overwrites
            enable_translucency=False, # defaults to False in performance mode
            enable_reflections=False, # defaults to False in performance mode
            antialiasing_mode="Off",
            dlss_mode="1", # defaults to 1 in performance mode
        )
            