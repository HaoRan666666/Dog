"""双轮足（只用两条后腿）平地行走训练环境配置。

任务描述：
    RP_wd 用两条后腿（LB/RB）+ 后轮支撑并行走，前腿（LF/RF）收起离地，
    形成「双轮足」倒立摆平衡任务（类似两轮自平衡 + 双腿支撑）。

与 ``walk_env_cfg``（四轮足平地走）的区别：
    - 动作只控制后腿（位置）+ 后轮（速度）+ 前腿（位置，用于收起）。
    - 观测/奖励/终止都围绕「在两条后轮上保持直立 + 追踪速度」设计。
"""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from Dog.robots.RP_wd import RP_wd_CFG

from . import mdp
from .walk_env_cfg import (
    SceneCfg as BaseSceneCfg,
    EventCfg as BaseEventCfg,
)

# ── 关节分组 ──────────────────────────────────────────────────────────
REAR_LEG_JOINTS = ["[L,R]B_ABAD_JOINT", "[L,R]B_HIP_JOINT", "[L,R]B_KENN_JOINT"]    # 后腿（支撑+平衡）
REAR_WHEEL_JOINTS = ["[L,R]B_FOOT_JOINT"]                                            # 后轮（驱动）
FRONT_LEG_JOINTS = ["[L,R]F_ABAD_JOINT", "[L,R]F_HIP_JOINT", "[L,R]F_KENN_JOINT"]   # 前腿（收起）
FRONT_WHEEL_JOINTS = ["[L,R]F_FOOT_JOINT"]                                           # 前轮（离地，不驱动）


# ── 场景配置 ──────────────────────────────────────────────────────────
@configclass
class SceneCfg(BaseSceneCfg):
    """复用平地场景（地形/灯光/接触传感器），只把机器人初始姿态改成「后腿站立」。"""

    robot = RP_wd_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            joint_pos={
                # 后腿立起支撑
                "[L,R]B_ABAD_JOINT": 0.0,
                "[L,R]B_HIP_JOINT": 1.0,        # TODO: 后腿站立角，需在 viewer 里调
                "[L,R]B_KENN_JOINT": -1.0,      # TODO: 同上
                # 前腿收起离地
                "[L,R]F_ABAD_JOINT": 0.0,
                "[L,R]F_HIP_JOINT": 2.2,        # TODO: 前腿折叠角，需在 viewer 里调
                "[L,R]F_KENN_JOINT": -2.5,      # TODO: 同上
            },
            joint_vel={".*": 0.0},
        ),
    )


# ── 观测配置 ──────────────────────────────────────────────────────────
@configclass
class ObservationsCfg:
    """观测：重心在姿态/角速度（倒立摆平衡必需）+ 指令 + 后腿/后轮状态。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=REAR_LEG_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=REAR_LEG_JOINTS + REAR_WHEEL_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """critic 特权观测：policy 基础上加线速度与高度扫描。"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=REAR_LEG_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=REAR_LEG_JOINTS + REAR_WHEEL_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ── 动作配置 ──────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    """动作：后腿位置（平衡）+ 后轮速度（前进/转向）+ 前腿位置（保持收起）。"""

    rear_legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=REAR_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    rear_wheels = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=REAR_WHEEL_JOINTS,
        scale=5.0,
        use_default_offset=False,
        preserve_order=True,
    )
    front_legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FRONT_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


# ── 指令配置 ──────────────────────────────────────────────────────────
@configclass
class CommandsCfg:
    """指令：只前后 + yaw（双轮无法横移），保留 heading 支持。"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.5, 1.5)
        ),
    )


# ── 课程配置 ──────────────────────────────────────────────────────────
@configclass
class CurriculumCfg:
    """课程：先低速站稳，再逐步放大速度指令。"""

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# ── 奖励配置 ──────────────────────────────────────────────────────────
@configclass
class RewardsCfg:
    """奖励：追踪速度 + 保持直立 + 常规平滑/力矩/触地惩罚。"""

    # ── 速度追踪 ──
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.8, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # ── 平衡（保持竖直）──
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)  # 俯仰/横滚偏离竖直即罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)             # 抑制俯仰/横滚角速度

    # ── 常规平滑惩罚 ──
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=REAR_LEG_JOINTS)},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    energy = RewTerm(
        func=mdp.energy, weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=REAR_LEG_JOINTS)},
    )

    # ── 触地惩罚：机身或前腿着地 = 没站住 ──
    base_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "[L,R]F_.*_LINK"]),
            "threshold": 1.0,
        },
    )


# ── 终止配置 ──────────────────────────────────────────────────────────
@configclass
class TerminationsCfg:
    """终止：超时 + 机身触地（倒立摆倒下）。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


# ── 事件配置 ──────────────────────────────────────────────────────────
@configclass
class EventCfg(BaseEventCfg):
    """事件：复用平地随机化，但关闭速度扰动（倒立摆初期不稳，禁推）。"""

    push_robot = None


# ── 环境（训练）───────────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Biped_Env(ManagerBasedRLEnvCfg):
    """RP_wd 双轮足（后腿站立）平地行走训练环境。"""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 8                       # 400Hz 物理 / 8 = 50Hz 控制
        self.episode_length_s = 20                # 平衡任务给更长时长
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation


# ── 环境（Play）───────────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Biped_Env_Play(RP_wd_Walk_Biped_Env):
    """双轮足 Play 环境（键盘遥控，单环境）。"""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.decimation = 8
        self.episode_length_s = 60
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.observations.policy.enable_corruption = False
        self.curriculum = None

        self.teleop_devices = DevicesCfg({
            "keyboard": Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=1.0,
                omega_z_sensitivity=2.0,
            ),
        })
