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
            lin_vel_x=(0.2, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
        # 平地：全向移动 + 旋转
        plane_ranges=mdp.TerrainSplitVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
        ),
        # num_cols=10, platform proportion=0.7 → 前7列是坑, 后3列是平地
        pit_col_threshold=7,
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
    """平台环境奖励：坑地形保持宽松约束（便于攀爬），平地恢复标准约束（保证全向步态质量）。

    通过 ``terrain_split_reward`` 对同一奖励项按地形类型应用不同 scale。
    """

    # ── 姿态约束 ──
    # dynamic_gravity_threshold=0.9:
    #   站直时 projected_gravity_z≈-1 → upright≈1 → plane_scale=-2.5（标准约束）
    #   爬台倾斜时 upright<0.9 → pit_scale=-0.1（允许大倾角）
    flat_orientation_l2 = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.flat_orientation_l2,
            "pit_scale": -0.1,
            "plane_scale": -2.5,
            "base_params": {},
            "dynamic_gravity_threshold": 0.95,
        },
    )

    # ── 统一的关节偏离（父类 joint_pos 的地形分权版本）──
    joint_pos = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.joint_position_penalty,
            "pit_scale": 0,
            "plane_scale": -0.3,
            "base_params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.3,
            },
            "dynamic_gravity_threshold": 0.95,
        },
    )
    # ── 大腿 ──
    joint_pos_hip = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.joint_position_penalty,
            "pit_scale": -0.01,
            "plane_scale": 0,
            "base_params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HIP_JOINT"]),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.3,
            },
            "dynamic_gravity_threshold": 0.95,
        },
    )
    # ── 小腿 ──
    joint_pos_kenn = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.joint_position_penalty,
            "pit_scale": -0.03,
            "plane_scale": 0,
            "base_params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_KENN_JOINT"]),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.3,
            },
            "dynamic_gravity_threshold": 0.95,
        },
    )
    # ── ABAD ──
    joint_pos_abad = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.joint_position_penalty,
            "pit_scale": -0.15,
            "plane_scale": 0,
            "base_params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ABAD_JOINT"]),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.1,
            },
            "dynamic_gravity_threshold": 0.95,
        },
    )

    # ── base 触地：坑 -10，平地 -20 ──
    base_contact_penalty = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.undesired_contacts,
            "pit_scale": 0,
            "plane_scale": -20.0,
            "base_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
                "threshold": 5.0,
            },
        },
    )
    # ── 关节限位 ──
    # 只限制外展和小腿（大腿放开，爬台需要大幅度前后摆动）
    joint_pos_limits = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.joint_pos_limits,
            "pit_scale": -10.0,
            "plane_scale": -20.0,
            "base_params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ABAD_JOINT"]),
            },
            "dynamic_gravity_threshold": 0.95,
        },
    )
    # ── 腿部触地：坑 -3（爬台时容易蹭到），平地 -10（标准约束）──
    leg_contact_penalty = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.undesired_contacts,
            "pit_scale": -1.0,
            "plane_scale": -20.0,
            "base_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*HIP_LINK", ".*KENN_LINK"]),
                "threshold": 1.0,
            },
        },
    )
    # ── 轮速惩罚：坑=0（允许蹭侧壁），平地=-0.05（引导用轮子而非伸腿横向移动）──
    wheel_vel_penalty = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.wheel_vel_penalty,
            "pit_scale": 0.0,
            "plane_scale": -0.05,
            "base_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*FOOT_LINK"]),
                "command_name": "base_velocity",
                "velocity_threshold": 0.3,
                "command_threshold": 0.06,
                "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
            },
        },
    )

    # ── 前轮离地惩罚：坑上强制前轮贴壁，平地不生效 ──
    # feet_air_time 本身 reward=(air_time - threshold) * first_contact，
    # 负的 pit_scale 把 reward 翻转为惩罚 → 前轮离地越久罚越重
    front_wheel_air = RewTerm(
        func=mdp.terrain_split_reward,
        weight=1.0,
        params={
            "base_func": mdp.feet_air_time,
            "pit_scale": -1.5,         # 坑上惩罚前轮离地
            "plane_scale": 0.0,        # 平地不生效
            "base_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LF_FOOT_LINK", "RF_FOOT_LINK"]),
                "command_name": "base_velocity",
                "threshold": 0.2,
            },
        },
    )

    # ── 动作平滑：抑制高频抖动 ──
    # 爬台时关节约束宽松，策略容易学会大幅快速摆腿，
    # 加强动作变化率和关节加速度惩罚以抑制迁移时的抖动。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)


# ── 平台事件配置 ──────────────────────────────────────────────────────
@configclass
class PlatformEventCfg(EventCfg):
    """平台事件：固定朝向和位置，确保机器人出生在地形格中心、面向前方。"""

    # 取消推动干扰（爬台时不需要外部扰动）
    push_robot = None

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

        # Play 模式保留坑洞 + 平地
        self.scene.terrain.terrain_generator.sub_terrains = {
            k: v for k, v in self.scene.terrain.terrain_generator.sub_terrains.items()
            if k in ("platform", "plane")
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
