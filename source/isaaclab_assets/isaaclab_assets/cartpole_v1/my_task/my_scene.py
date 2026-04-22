import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import CARTPOLE_ROBOT_CFG

@configclass
class Cartpole_v1_SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, -2.0),
        # ),
        spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


    # articulation
    cartpole: ArticulationCfg = CARTPOLE_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
