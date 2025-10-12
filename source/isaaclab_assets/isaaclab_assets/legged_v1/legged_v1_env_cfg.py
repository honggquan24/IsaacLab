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


@configclass
class LeggedRobotV1SceneConfig(InteractiveSceneCfg):
    # add light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # add robot 
    robot: Articulation = LEGGED_ROBOT_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    print(robot)

    # add sensor /World/envs/env_0/Robot/Robot_2_leg_base/robot_2_leg/robot_2_leg/Group_1/base/base
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot_2_leg_base/robot_2_leg/robot_2_leg/Group_1/base/Camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=[
            "rgb",
            # "distance_to_camera",
            # "distance_to_image_plane",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.25, 0.25, 0.5),
            rot=tuple(
                math_utils.quat_from_euler_xyz(
                    torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
                )[0].tolist()
            ),
            convention="ros"),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot_2_leg_base/robot_2_leg/robot_2_leg/Group_1",
        update_period=0.1,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 20.0)
        ),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[1.6, 1.0],
        ),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot_2_leg_base/robot_2_leg/robot_2_leg/Group_1",
        update_period=0.1,
        history_length=6,
        gravity_bias=(0, 0, 0),
        debug_vis=True
    )


@configclass
class ActionCfg:
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
        scale=100.0
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos_rel = ObservationTermCfg(func=observations.joint_pos_rel)
        joint_vel_rel = ObservationTermCfg(func=observations.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:

    reset_hip_left_position = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["Revolute_1"]
            ),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-math.pi, math.pi),
        }
    )

    reset_hip_right_position = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["Revolute_2"]
            ),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-math.pi, math.pi),
        }
    )

    reset_knee_left_position = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["Revolute_3"]
            ),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-math.pi, math.pi),
        }
    )

    reset_knee_right_position = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["Revolute_4"]
            ),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-math.pi, math.pi),
        }
    )

    reset_ankle_left_position = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["Revolute_5"]
            ),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-math.pi, math.pi),
        }
    )

    reset_ankle_right_position = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["Revolute_6"]
            ),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-math.pi, math.pi),
        }
    )

@configclass
class RewardCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewardTermCfg(func=rewards.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewardTermCfg(func=rewards.is_terminated, weight=-1.0)

    # (3) Primary task: keep pole upright

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = TerminationTermCfg(func=terminations.time_out, time_out= True)

    # (2) Robot out of bounds

@configclass
class LeggedRobotV1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the robot environment."""

    # Scene settings
    scene: LeggedRobotV1SceneConfig = LeggedRobotV1SceneConfig(
        num_envs=2,
        env_spacing=4.0, 
        clone_in_fabric=True
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
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation



