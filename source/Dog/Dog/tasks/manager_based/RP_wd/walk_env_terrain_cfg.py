import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from Dog.assets.terrain.step_terrain import STEP_TERRAINS_CFG

from . import mdp
from .walk_env_cfg import (
    RP_wd_Walk_Flat_Env,
    SceneCfg,
    ObservationsCfg,
    CurriculumCfg,
)
LEG_JOINTS = [".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
WHEEL_JOINTS = [".*_FOOT_JOINT"]

@configclass
class TerrainSceneCfg(SceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STEP_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class TerrainObservationsCfg(ObservationsCfg):
    # PolicyCfg 直接继承平地配置，不加 height_scan（实机无法观测）

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """critic 特权观测：平地配置 + 高度扫描"""
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TerrainCommandsCfg:
    """地形环境的指令配置：低速行走，便于在复杂地形上学习。"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class TerrainCurriculumCfg(CurriculumCfg):
    """地形课程：只保留地形难度课程，取消速度课程。"""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None


@configclass
class RP_wd_Walk_Terrain_Env(RP_wd_Walk_Flat_Env):
    """地形环境配置：继承平地环境，替换场景、观测和课程配置。"""
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=2048, env_spacing=4.0)
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    commands: TerrainCommandsCfg = TerrainCommandsCfg()
    curriculum = TerrainCurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 8
        self.episode_length_s = 20
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
