"""高台（平台）地形训练环境配置。

该模块定义了 RP_wd 轮腿机器人在高台/箱体地形上的强化学习训练环境。
机器人需要学习攀爬 0.3m~0.8m 高的箱体平台，并通过课程学习逐步增加难度。
"""

import isaaclab.sim as sim_utils
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from Dog.assets.terrain.platform_terrain import PLATFORM_TERRAINS_CFG

from . import mdp
from .walk_env_cfg import (
    RP_wd_Walk_Flat_Env,
    SceneCfg,
    ObservationsCfg,
    CurriculumCfg,
    RewardsCfg,
    EventCfg,
)

LEG_JOINTS = [".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
WHEEL_JOINTS = [".*_FOOT_JOINT"]

# ── 平台场景配置 ──────────────────────────────────────────────────────
@configclass
class PlatformSceneCfg(SceneCfg):
    """高台/平台地形场景：箱体 + 平面混合地形。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PLATFORM_TERRAINS_CFG,
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


# ── 平台观测配置 ──────────────────────────────────────────────────────
@configclass
class PlatformObservationsCfg(ObservationsCfg):
    """平台环境观测：Policy 无特权观测（与实机一致），Critic 加入高度扫描。"""

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """critic 特权观测：平地配置 + 高度扫描（用于感知前方平台高度）。"""
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ── 平台指令配置 ──────────────────────────────────────────────────────
@configclass
class PlatformCommandsCfg:
    """平台指令：坑地形只前进，平地全向移动。根据 env 所处地形类型自动切换。"""

    base_velocity = mdp.TerrainSplitVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.001,
        rel_heading_envs=1.0,          # 平地全部开启 heading
        heading_command=True,           # 开启 heading（仅平地生效）
        heading_control_stiffness=0.5,
        debug_vis=True,
        # 坑地形：只前进，无旋转
        pit_ranges=mdp.TerrainSplitVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.8), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
        # 平地：全向移动 + 旋转
        plane_ranges=mdp.TerrainSplitVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
        ),
        # num_cols=10, platform proportion=0.8 → 前8列是坑, 后2列是平地
        pit_col_threshold=8,
    )


# ── 平台课程配置 ──────────────────────────────────────────────────────
@configclass
class PlatformCurriculumCfg(CurriculumCfg):
    """平台课程：地形难度（箱体高度）课程，取消速度课程。"""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = None


# ── 平台奖励配置 ──────────────────────────────────────────────────────
@configclass
class PlatformRewardsCfg(RewardsCfg):
    """平台环境奖励：保留基础姿态约束，调整爬台相关惩罚权重。"""

    # ── 降低姿态约束权重：攀爬时身体需要更大的倾角 ──
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, weight=-0.1  # 平地 -2.5, 地形 -0.2
    )

    # ── 关掉父类的合并项（joint_pos 包含 ABAD+HIP+KENN）──
    joint_pos = None
    # ── 大腿和小腿分开约束：大腿放宽，小腿保留 ──
    joint_pos_hip = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.01,  # 大腿极轻约束
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HIP_JOINT"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    joint_pos_kenn = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_KENN_JOINT"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    # ABAD 单独加重惩罚，防止外展抬腿
    joint_pos_abad = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ABAD_JOINT"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # ── 加重 base 触地惩罚：爬坑时更不容忍翻倒 ──
    base_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,  # 平地 -20.0 → 翻倍
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), "threshold": 5.0},
    )
    # ── 放宽关节限位惩罚：爬坑需要更大的关节运动范围 ──
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0,  # 平地 -20.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
    )
    # ── 降低腿部触地惩罚：爬台时大腿容易蹭到平台边缘 ──
    leg_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,  # 平地 -10.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*HIP_LINK", ".*KENN_LINK"]), "threshold": 1.0},
    )
    # 关掉轮速惩罚：允许前轮蹭侧壁辅助攀爬
    wheel_vel_penalty = None


# ── 平台事件配置 ──────────────────────────────────────────────────────
@configclass
class PlatformEventCfg(EventCfg):
    """平台事件：固定朝向和位置，确保机器人出生在地形格中心、面向前方。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


# ── 平台环境（训练）───────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Platform_Env(RP_wd_Walk_Flat_Env):
    """RP_wd 轮腿机器人高台平台行走训练环境。

    继承平地环境，替换为平台地形、适配观测/指令/课程/奖励。
    """
    scene: PlatformSceneCfg = PlatformSceneCfg(num_envs=2048, env_spacing=4.0)
    observations: PlatformObservationsCfg = PlatformObservationsCfg()
    commands: PlatformCommandsCfg = PlatformCommandsCfg()
    curriculum = PlatformCurriculumCfg()
    rewards: PlatformRewardsCfg = PlatformRewardsCfg()
    events: PlatformEventCfg = PlatformEventCfg()

    def __post_init__(self) -> None:
        self.decimation = 8
        self.episode_length_s = 10  # 比地形(8s)稍长，爬台需要更多时间
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation


# ── 平台环境（Play）───────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Platform_Env_Play(RP_wd_Walk_Platform_Env):
    """RP_wd 轮腿机器人高台平台行走 Play 环境（键盘遥控）。"""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.decimation = 8
        self.episode_length_s = 40
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.observations.policy.enable_corruption = False

        # 从低难度开始（避免出生在高台阶上）
        self.scene.terrain.max_init_terrain_level = 0

        # Play 模式保留坑洞 + 台阶 + 平地
        self.scene.terrain.terrain_generator.sub_terrains = {
            k: v for k, v in self.scene.terrain.terrain_generator.sub_terrains.items()
            if k in ("platform", "stairs", "plane")
        }

        self.teleop_devices = DevicesCfg({
            "keyboard": Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=1.0,
                omega_z_sensitivity=2.0,
            ),
        })

        self.curriculum = None
        # 关闭 base 接触终止，爬台时容易蹭到平台边缘
        self.terminations.base_contact = None
