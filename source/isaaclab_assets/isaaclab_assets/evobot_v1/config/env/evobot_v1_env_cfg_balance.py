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

    # Add IMU sensor - mounted on base_link (main body)
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/evobot/base_link",
        update_period=0.02,  # Changed from 0.1 to match control frequency (50Hz)
        gravity_bias=(0.0, 0.0, 0.0),
    )

    # Height scanner - mounted on head_link for forward scanning
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/evobot/base_link",
        update_period=0.01,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[0.3, 0.3], # type: ignore
        ),
        mesh_prim_paths=["/World/Ground"],
    )

    # Contact sensors for wheels
    contact_forces_wheel_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/evobot/left_wheel",
        update_period=0.01,
    )

    contact_forces_wheel_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/evobot/right_wheel",
        update_period=0.01,
    )


@configclass
class ActionCfg:
    """Action configuration for joint effort control."""

    joint_effort = actions.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
        'arm_joint', 
        'left_wheel_joint',
        'right_wheel_joint', 
        'base_joint',
        'left_grabbing_joint',
        'right_grabbing_joint', 
        'right_levers_2_joint',
        'left_levers_2_joint', 
        'left_turntable_joint', 
        'left_levers_joint_1',
        'right_turntable_joint',
        'right_levers_joint_1'],
        scale=500.0,
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
                "z": (0.72, 0.72),
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
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # (6) Angular velocity stability (roll/pitch rates)
    ang_vel_xy = RewardTermCfg(
        func=ang_vel_xy_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # (7) Base height tracking
    base_height = RewardTermCfg(
        func=base_height_l2,
        weight=-10.0,
        params={
            "target_height": 0.70,  # Target upright standing height
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
    # base_height = TerminationTermCfg(
    #     func=terminations.root_height_below_minimum,
    #     params={
    #         "minimum_height": 0.25,
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    # BAD ORIENTATION
    # bad_orientation = TerminationTermCfg(
    #     func=terminations.bad_orientation,
    #     params={
    #         "limit_angle": math.pi / 3,  # 60° tolerance
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )


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

        self.decimation = 1  # Control freq = 60/1 = 60 Hz
        self.episode_length_s = 20  # Episode duration

        # Viewer settings
        self.viewer.eye = (5.0, 0.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # FIX: Added lookat point

        # Simulation settings
        self.sim.dt = 1 / 60  # Physics timestep = 60 Hz
        self.sim.render_interval = self.decimation  # Render every decimation steps
