"""Navigation NÉ VẬT CẢN bằng terrain sinh thủ tục (procedural obstacles).

Vì RayCaster bản này CHỈ đọc 1 mesh (không thấy kho USD đa-mesh), ta thay kho
bằng TERRAIN GENERATOR sinh hộp vật cản ngẫu nhiên. Toàn bộ tile + vật cản được
gộp thành MỘT mesh `/World/ground` duy nhất → LiDAR (RayCaster) quét THẤY ĐẦY ĐỦ
→ train SONG SONG nhiều env, học né thật. Mỗi tile có `platform_width` trống ở
tâm → robot spawn đúng chỗ sạch (giải luôn "spawn né đồ").

⚠️ num_envs nên ≤ num_rows × num_cols (= số tile) để mỗi env một tile, tránh 2
robot chồng nhau. Mặc định lưới 32×32 = 1024 tile → train tốt với --num_envs 1024.
Muốn 4096 thì nâng lưới lên 64×64.

Train:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Legged-V5-Obstacle-Nav --num_envs 1024 --headless
Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Legged-V5-Obstacle-Nav --num_envs 16
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass

from ..legged_v5_cfg import LEGGED_V5_CFG
from .legged_v5_navigation_env_cfg import LeggedV5NavigationEnvCfg, NavSceneCfg
from .legged_v5_warehouse_nav_env_cfg import (
    LIDAR_HORIZONTAL_RES,
    LIDAR_MAX_DISTANCE,
    WarehouseObsCfg,
    WarehouseRewardCfg,
    WarehouseTermCfg,
)

# ─────────────────────────── Terrain sinh vật cản ─────────────────────────────
# CHỈ vật cản LỒI LÊN (mode="fixed" → height dương; "choice" sinh cả hố ÂM → lõm).
# Cao 0.8–1.4 m (CAO HƠN LiDAR ở ~0.65 m = base z≈0.35 + offset 0.3) → tia luôn trúng.
# platform_width 2.5 m: vùng trống ở tâm tile cho robot spawn + có chỗ xoay.
OBSTACLE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=1.0,
    num_rows=32,
    num_cols=32,          # 32×32 = 1024 tile (một env một tile)
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            num_obstacles=5,
            obstacle_height_mode="fixed",   # CHỈ cột dương (lồi lên), KHÔNG hố lõm
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.8, 1.4),  # cao theo difficulty từng tile, đều > tầm LiDAR
            platform_width=2.5,
        ),
    },
)


# ─────────────────────────── Scene (terrain obstacle + LiDAR) ─────────────────

@configclass
class ObstacleNavSceneCfg(NavSceneCfg):
    """Thay plane bằng terrain sinh vật cản (1 mesh) + LiDAR 2D."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBSTACLE_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.3,
        ),
        debug_vis=False,
    )

    robot: Articulation = LEGGED_V5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # LiDAR 2D quét mesh terrain (gồm vật cản) — cùng cấu hình bản warehouse.
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
        debug_vis=False,
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class LeggedV5ObstacleNavEnvCfg(LeggedV5NavigationEnvCfg):
    """Nav né vật cản trên terrain sinh thủ tục — train song song được."""

    scene:        ObstacleNavSceneCfg = ObstacleNavSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: WarehouseObsCfg     = WarehouseObsCfg()
    rewards:      WarehouseRewardCfg  = WarehouseRewardCfg()
    terminations: WarehouseTermCfg    = WarehouseTermCfg()

    def __post_init__(self):
        super().__post_init__()

        # Đích trong phạm vi 1 tile (±2.5 m) → robot phải luồn quanh vật cản tới đích.
        self.commands.goal.ranges.pos_x = (-2.5, 2.5)
        self.commands.goal.ranges.pos_y = (-2.5, 2.5)

        # Spawn quanh TÂM tile (platform trống) — offset nhỏ, KHÔNG yaw (yaw làm robot lật).
        self.events.reset_position.params["pose_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5),
        }
