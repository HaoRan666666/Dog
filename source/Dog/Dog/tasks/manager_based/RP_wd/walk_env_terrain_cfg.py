import isaaclab.sim as sim_utils
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import math

from Dog.assets.terrain.step_terrain import STEP_TERRAINS_CFG

from . import mdp
from .walk_env_cfg import (
    RP_wd_Walk_Flat_Env,
    SceneCfg,
    ObservationsCfg,
    CurriculumCfg,
    RewardsCfg,
)
LEG_JOINTS = [".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
WHEEL_JOINTS = [".*_FOOT_JOINT"]

@configclass
class TerrainSceneCfg(SceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STEP_TERRAINS_CFG,
        max_init_terrain_level=0,
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
    """地形环境的指令配置：加入 yaw 控制，提升复杂地形上的转向能力。"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-0.5, 1.0), ang_vel_z=(-1, 1), heading=(-math.pi, math.pi)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-0.5, 1.0), ang_vel_z=(-1, 1)
        ),
    )


@configclass
class TerrainCurriculumCfg(CurriculumCfg):
    """地形课程：只保留地形难度课程，取消速度课程。"""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None  # 预训练后不需要速度课程，直接固定速度


@configclass
class TerrainRewardsCfg(RewardsCfg):
    """地形奖励：移除不利于爬坡/上台阶的惩罚项。"""

    # ── 降权重：保留基础姿态约束，防止"前倾滑行"作弊 ──
    flat_orientation_l2 = None

    joint_pos=None

    # ── 降权重：台阶上允许更大关节偏移，但要保留一定约束 ──

    # ABAD 单独加重惩罚，防止外展抬腿
    joint_pos_abad = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ABAD_JOINT"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # ── 足端步态奖励：暂移除（地形爬坡不需要正常踏步步态），后续需要再加 ──
    feet_air_time = None

@configclass
class RP_wd_Walk_Terrain_Env(RP_wd_Walk_Flat_Env):
    """地形环境配置：继承平地环境，替换场景、观测和课程配置。"""
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=2048, env_spacing=4.0)
    observations: TerrainObservationsCfg = TerrainObservationsCfg()
    commands: TerrainCommandsCfg = TerrainCommandsCfg()
    curriculum = TerrainCurriculumCfg()                                                                     
    rewards: TerrainRewardsCfg = TerrainRewardsCfg()

    def __post_init__(self) -> None:
        self.decimation = 8
        self.episode_length_s = 8
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation


@configclass
class RP_wd_Walk_Terrain_Env_Play(RP_wd_Walk_Terrain_Env):

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.decimation = 8
        self.episode_length_s = 40
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.observations.policy.enable_corruption = False

        # 从中等难度（20级）开始
        self.scene.terrain.max_init_terrain_level = 15
        # Play 模式只用倒金字塔（下台阶）
        self.scene.terrain.terrain_generator.sub_terrains = {
            k: v for k, v in self.scene.terrain.terrain_generator.sub_terrains.items()
            if "inv" in k
        }

        self.teleop_devices = DevicesCfg({
            "keyboard": Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=1.0,
                omega_z_sensitivity=2.0,
            ),
        })

        self.curriculum = None
        self.terminations.base_contact = None  # 台阶上 base 易蹭台阶，关掉接触终止
