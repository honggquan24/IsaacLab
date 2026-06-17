"""Navigation trong kho (warehouse) tránh vật cản bằng LiDAR — robot V5.

Kế thừa LeggedV5NavigationEnvCfg (nav mặt phẳng) và bổ sung:
  - Kho mô phỏng: Simple_Warehouse (USD nucleus), nạp làm TERRAIN tại
    /World/ground (1 mesh tĩnh, dùng chung mọi env).
  - Cảm biến LiDAR 2D: RayCasterCfg + LidarPatternCfg (360°, 1 kênh).
  - Quan sát: thêm các tia khoảng cách LiDAR vào policy tầng cao.
  - Phần thưởng/kết thúc: phạt khi lại gần vật cản, kết thúc khi sắp đâm.

⚠️ GIỚI HẠN RayCaster (bản IsaacLab trong workspace này):
  RayCaster chỉ hỗ trợ MỘT mesh và đọc mesh con ĐẦU TIÊN dưới
  `mesh_prim_paths`. Warehouse là asset nhiều mesh nên LiDAR có thể chỉ
  "thấy" một phần hình học. Hai cách để LiDAR thấy đầy đủ vật cản:
    (a) Nâng cấp lên MultiMeshRayCaster (IsaacLab mới) — rồi đổi RayCasterCfg
        thành MultiMeshRayCasterCfg với merge_prim_meshes=True.
    (b) Gộp collision của kho thành 1 mesh (Blender/USD) rồi trỏ
        mesh_prim_paths vào mesh đó.
  Vì mesh dùng chung world-coords nên cấu hình này chạy đúng nhất với
  num_envs=1 (demo/inference). Train song song cần phương án (a).

  Robot vẫn VA CHẠM vật lý với kho (collider của warehouse) bất kể LiDAR —
  nên termination dựa trên LiDAR là lớp an toàn bổ sung, không phải duy nhất.

Tầng dưới (locomotion) vẫn là policy.pt đã đóng băng như nav phẳng — phải set
`policy_path` trong file env nav gốc trước khi train/chạy.

Play (1 robot, xem LiDAR):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V5-Warehouse-Nav --num_envs 1
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import (
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import sensors as nav_sensors
from .legged_v5_navigation_env_cfg import (
    LeggedV5NavigationEnvCfg,
    NavObsCfg,
    NavRewardCfg,
    NavSceneCfg,
    NavTermCfg,
)

WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"  # nhiều kệ/đồ hơn warehouse.usd

# Số tia = horizontal_fov / horizontal_res = 360 / 4 = 90.
LIDAR_HORIZONTAL_RES = 4.0
LIDAR_MAX_DISTANCE = 10.0


# ─────────────────────────── Scene (kho làm terrain + LiDAR) ──────────────────

@configclass
class WarehouseNavSceneCfg(NavSceneCfg):
    """Thay mặt phẳng bằng kho warehouse (terrain USD) + cảm biến LiDAR 2D."""

    # Kho làm terrain — 1 mesh tĩnh global tại /World/ground (thay plane của base).
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=WAREHOUSE_USD,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.3,
        ),
        debug_vis=False,
    )

    # LiDAR 2D gắn trên thân robot, quét ngang 360° vào mesh kho.
    #   prim_path: body 'base' của V5 (USD lồng Robot/Robot/Robot — khớp contact sensor).
    #              Nếu sai, đối chiếu scene.robot.body_names hoặc Stage panel.
    #   ray_alignment="yaw": tia nằm ngang, xoay theo hướng thân. Nếu scan bị
    #              nghiêng do thân xoay 90° quanh X, đổi sang "world".
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        ray_alignment="yaw",
        max_distance=LIDAR_MAX_DISTANCE,
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=LIDAR_HORIZONTAL_RES,
        ),
        debug_vis=False,   # bật True chỉ khi play/xem; train để False (viz 0 tia → crash)
    )


# ─────────────────────────── Observation (+ LiDAR) ────────────────────────────

@configclass
class WarehousePolicyCfg(NavObsCfg.PolicyCfg):
    """Obs tầng cao + tia khoảng cách LiDAR (chuẩn hoá [0,1])."""

    lidar_scan = ObservationTermCfg(
        func=nav_sensors.lidar_ranges,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": LIDAR_MAX_DISTANCE},
    )


@configclass
class WarehouseObsCfg(NavObsCfg):
    policy: WarehousePolicyCfg = WarehousePolicyCfg()


# ─────────────────────────── Reward (+ tránh vật cản) ─────────────────────────

@configclass
class WarehouseRewardCfg(NavRewardCfg):
    obstacle_avoidance = RewardTermCfg(
        func=nav_sensors.obstacle_proximity_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "safe_distance": 0.6,
            "max_distance": LIDAR_MAX_DISTANCE,
        },
    )


# ─────────────────────────── Termination (+ va chạm) ──────────────────────────

@configclass
class WarehouseTermCfg(NavTermCfg):
    collision = TerminationTermCfg(
        func=nav_sensors.too_close_to_obstacle,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "min_distance": 0.25,
            "max_distance": LIDAR_MAX_DISTANCE,
        },
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV5WarehouseNavEnvCfg(LeggedV5NavigationEnvCfg):

    # Mesh LiDAR dùng chung world-coords → num_envs=1 cho demo/inference.
    scene:        WarehouseNavSceneCfg = WarehouseNavSceneCfg(num_envs=1, env_spacing=20.0)
    observations: WarehouseObsCfg      = WarehouseObsCfg()
    rewards:      WarehouseRewardCfg   = WarehouseRewardCfg()
    terminations: WarehouseTermCfg     = WarehouseTermCfg()

    def __post_init__(self):
        super().__post_init__()

        # Đích nằm trong lòng kho — tinh chỉnh theo kích thước warehouse.usd thực tế.
        self.commands.goal.ranges.pos_x = (-5.0, 5.0)
        self.commands.goal.ranges.pos_y = (-3.0, 3.0)

        # Spawn robot ở lối đi trống gần tâm kho (tránh sinh vào trong kệ/tường).
        self.events.reset_position.params["pose_range"] = {
            "x": (-1.0, 1.0), "y": (-1.0, 1.0),   # KHÔNG yaw (yaw làm robot lật do base xoay 90°X)
        }

        self.viewer.eye    = (0.0, -8.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
