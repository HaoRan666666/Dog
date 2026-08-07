"""台阶 + 翻倒自救训练环境配置。

该模块在台阶地形训练的基础上，增加翻倒后自主恢复站立的能力。
机器人以一定概率从侧倒/后倒姿态出生，通过重力对齐奖励引导其学会翻身站起，
同时保持台阶跨越能力不退化。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
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


# ── 场景配置 ──────────────────────────────────────────────────────────
@configclass
class StairFallSceneCfg(SceneCfg):
    """台阶地形场景。"""

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


# ── 观测配置 ──────────────────────────────────────────────────────────
@configclass
class StairFallObservationsCfg(ObservationsCfg):
    """Critic 特权观测：加入高度扫描。"""

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ── 指令配置 ──────────────────────────────────────────────────────────
@configclass
class StairFallCommandsCfg:
    """台阶地形指令：低速全向移动 + yaw 控制。"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8), lin_vel_y=(-0.4, 0.6),
            ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8), lin_vel_y=(-0.4, 0.6),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


# ── 课程配置 ──────────────────────────────────────────────────────────
@configclass
class StairFallCurriculumCfg(CurriculumCfg):
    """台阶地形课程 + 速度课程。"""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# ── 奖励配置 ──────────────────────────────────────────────────────────
@configclass
class StairFallRewardsCfg:
    """台阶 + 自救奖励。

    核心改动：大幅加强重力对齐奖励（flat_orientation_l2），
    作为翻倒后恢复站立的主要引导信号。
    """

    # ── 任务奖励 ──
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.8,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ── 常规惩罚 ──
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.03)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    energy = RewTerm(
        func=mdp.energy, weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )

    # ── 核心：重力对齐奖励（翻倒恢复的主信号）──
    # 体重力投影 g_xy = [g_x, g_y]，站立时 ≈ [0, 0]，侧倒时值很大
    # weight=-1.5 比原台阶 (-0.2) 强 7.5 倍，确保恢复信号足够强
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, weight=-1.5,
    )

    # ── 侧倾专项惩罚（防侧翻 + 侧倒恢复辅助）──
    side_tilt_l2 = RewTerm(
        func=mdp.side_tilt_l2, weight=-2.0,
    )

    # ── 触地惩罚 ──
    # base 触地：翻倒时必然触发，降低惩罚避免压倒恢复信号
    base_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
                "threshold": 1.0},
    )
    # 腿触地：爬台阶允许适度蹭到
    leg_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=[".*HIP_LINK", ".*_KENN_LINK"]), "threshold": 1.0},
    )

    # ── 关节约束 ──
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    # 整体关节偏离
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty, weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    # ABAD 专项约束
    joint_pos_abad = RewTerm(
        func=mdp.joint_position_penalty, weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ABAD_JOINT"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # ── 静止站立 ──
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1, weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        },
    )

    # ── 轮速惩罚 ──
    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty, weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK"),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*FOOT_JOINT"),
        },
    )


# ── 终止配置 ──────────────────────────────────────────────────────────
@configclass
class StairFallTerminationsCfg:
    """放宽终止条件：去掉 base_contact，给翻倒恢复留时间。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 侧翻 150° / 后倒 60° 才终止（接近完全倒扣，无法恢复）
    side_tilt = DoneTerm(
        func=mdp.side_tilt,
        params={"limit_angle": math.radians(150.0),
                "backward_angle": math.radians(60.0)},
    )


# ── 事件配置 ──────────────────────────────────────────────────────────
@configclass
class StairFallEventCfg:
    """启动 + 重置 + 周期事件。

    核心改动：reset_base 的 roll/pitch 扩宽到摔倒范围，
    利用均匀采样让 ~40% 的 episode 从摔倒姿态出生。
    """

    # ── startup ──
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 2.0),
            "dynamic_friction_range": (0.15, 1.5),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    leg_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="[L,R][F,B]_.*_LINK"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (-0.05, 0.05)},
        },
    )

    scale_leg_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    scale_wheel_actuator_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    scale_leg_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    scale_wheel_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    # ── reset ──
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # 核心：扩宽 roll/pitch 范围到摔倒姿态
    # roll ~ U(-2.0, 2.0):  |roll|<0.5 正常 → 25% | 0.5<|roll|<1.0 倾斜 → 25% | |roll|>1.0 侧倒 → 50%
    # pitch ~ U(-1.5, 0.5): pitch<-0.5 后倒 → 33% | pitch>-0.5 正常→ 67%
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-2.0, 2.0),       # ±115° 侧倒范围
                "pitch": (-1.5, 0.5),       # -86°~29° 后倒到前倾
                "yaw": (-3.14, 3.14),
            },
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ── interval ──
    # 保留随机推力扰动（模拟外力导致摔倒的场景）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0),
                                   "yaw": (-2.0, 2.0)}},
    )


# ═══════════════════════════════════════════════════════════════════════
# 训练环境
# ═══════════════════════════════════════════════════════════════════════
@configclass
class RP_wd_Walk_Stair_Fall_Env(RP_wd_Walk_Flat_Env):
    """RP_wd 台阶 + 翻倒自救训练环境。

    继承平地环境，替换为台阶地形，加入翻倒姿态出生和恢复奖励。
    训练目标：保持台阶跨越能力的同时，学会从侧倒/后倒姿态自主恢复站立。
    """

    scene: StairFallSceneCfg = StairFallSceneCfg(num_envs=2048, env_spacing=4.0)
    observations: StairFallObservationsCfg = StairFallObservationsCfg()
    commands: StairFallCommandsCfg = StairFallCommandsCfg()
    curriculum = StairFallCurriculumCfg()
    rewards: StairFallRewardsCfg = StairFallRewardsCfg()
    terminations: StairFallTerminationsCfg = StairFallTerminationsCfg()
    events: StairFallEventCfg = StairFallEventCfg()

    def __post_init__(self) -> None:
        self.decimation = 8
        self.episode_length_s = 12  # 比平地 (12s) 不变，但比台阶 (8s) 长 50%
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation


# ═══════════════════════════════════════════════════════════════════════
# Play 环境（键盘遥控测试）
# ═══════════════════════════════════════════════════════════════════════
@configclass
class RP_wd_Walk_Stair_Fall_Env_Play(RP_wd_Walk_Stair_Fall_Env):
    """台阶 + 自救 Play 环境（键盘遥控）。"""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.decimation = 8
        self.episode_length_s = 40
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.observations.policy.enable_corruption = False

        # 从中等难度开始
        self.scene.terrain.max_init_terrain_level = 15
        # Play 只用倒金字塔（下台阶）
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
