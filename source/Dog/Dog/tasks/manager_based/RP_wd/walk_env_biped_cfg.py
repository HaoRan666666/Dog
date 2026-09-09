"""双轮足（只用两条后腿）平地行走训练环境配置。

任务描述：
    RP_wd 从与平地完全相同的四腿站立姿态出发，仅通过奖励学习用两条后腿（LB/RB）
    + 后轮支撑并行走，前腿（LF/RF）抬起离地，形成「双轮足」倒立摆平衡任务。

与 ``walk_env_cfg``（四轮足平地走）的区别（只改命令 + 奖励）：
    - 动作 / 观测 / 事件 / 终止 / 课程 / 初始姿态 全部与平地一致。
    - 命令：关闭侧移（lin_vel_y=0），保留前进速度与 yaw 角速度（转向）；heading 关闭。
    - 多一条「前腿 + 前轮触地惩罚」，逼策略把前腿抬起来、后腿站立。
    - 机身朝向由「机身水平」改为「机身竖直」（upright_orientation_l2），
      水平要求（flat_orientation_l2）置 None（移除）。
    - 移除机体系速度追踪与抬腿奖励（track_lin_vel_xy_exp / track_ang_vel_z_exp /
      lin_vel_z_l2 / feet_air_time / feet_clearance / leg_usage_balance 置 None），
      前进追踪改用沿朝向的 track_lin_vel_heading_exp、转向改用世界系 track_ang_vel_w_z_exp。
"""

from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .walk_env_cfg import (
    RP_wd_Walk_Flat_Env,
    RewardsCfg as FlatRewardsCfg,
)


# ── 命令配置（双轮足：前进/后退 + 转向，无侧移）────────────────────────
@configclass
class BipedCommandsCfg:
    """双轮足速度命令：前向速度 + yaw 角速度，无侧移。

    双轮足站起后是「差速驱动」：左右后轮（LB/RB）各自独立转动，转速差产生 yaw，
    所以保留 ang_vel_z（转向），只关掉侧移 lin_vel_y。heading_command 关闭，
    因为标准 heading_w 由 body-x 的投影算得，站起后 body-x 朝上会退化成 atan2(0,0)。
    lin_vel_x 采样后由 ``track_lin_vel_heading_exp``（沿朝向）与 ``track_ang_vel_w_z_exp``
    （世界系 yaw）追踪。
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        heading_command=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.5, 2.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.5, 1.5),
        ),
    )


# ── 奖励配置（平地奖励 + 前腿触地惩罚 + 机身竖直）─────────────────────
@configclass
class BipedRewardsCfg(FlatRewardsCfg):
    """奖励：继承平地，但把机体系「抬腿/转向/竖直速度」相关项全部移除（置 None），换成双轮足自平衡所需。"""

    # 前腿任意连杆（ABAD/HIP/KENN/FOOT）触地 → 惩罚。
    # 四腿站立时前轮持续着地，会一直吃到这条惩罚；策略要消除它只能把前腿抬起来，
    # 用两条后腿 + 后轮支撑行走。权重降到 -5（原 -20 从 episode 一开始就重罚天然触地的
    # 前脚，逼前腿在后腿还没站稳时就急速甩开），留给 front_hip_tuck/front_kenn_tuck
    # 的站立门控去把「抬前腿」和「站起来」两个时序解耦。
    front_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="[L,R]F_.*_LINK"),
            "threshold": 1.0,
        },
    )

    # 机身朝向：平地要求「机身水平」（flat_orientation_l2 罚 g_x²+g_y²，机身 Z 轴朝上）。
    # 双轮足站立要求「机身竖直」（罚 g_y²+g_z²，机身 X 轴朝上），故把继承来的水平要求置 None（移除），
    # 换成竖直朝向奖励。upright_orientation_l2 对 g_x 符号对称，上下方向由前腿/机身触地惩罚打破。
    flat_orientation_l2 = None
    upright_orientation_l2 = RewTerm(func=mdp.upright_orientation_l2, weight=-2.5)

    # ── 机体系速度追踪 / 抬腿奖励：双轮足全部移除（None）───────────────
    # 前进追踪：由沿朝向的 track_lin_vel_heading_exp 替代（机体系 90° 后仰后 body x-y 已错位）。
    track_lin_vel_xy_exp = None
    # yaw 追踪：由世界系 track_ang_vel_w_z_exp 替代（后仰后 body-z 已非竖直）。
    track_ang_vel_z_exp = None
    # 竖直速度惩罚：后仰后 body-z = 前进方向，这条会罚前进，移除。
    lin_vel_z_l2 = None

    # 三条抬腿奖励：双轮足只靠轮子，不抬腿，全部移除。
    feet_air_time = None
    feet_clearance = None
    leg_usage_balance = None
    # 足端落地冲击速度：双轮足不抬腿（无下放），移除。
    feet_landing_impact = None

    # 沿朝向的前向速度追踪：命令 [:0] 视为「沿机身面朝方向」的速度。
    track_lin_vel_heading_exp = RewTerm(
        func=mdp.track_lin_vel_heading_exp,
        weight=1.8,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 世界系 yaw 角速度追踪：命令 [:2] 视为「绕世界 z 的转向角速度」。
    track_ang_vel_w_z_exp = RewTerm(
        func=mdp.track_ang_vel_w_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 关节位置超限惩罚：平地版本对 LEG_JOINTS（含大腿 HIP）惩罚；双轮足站立时
    # 大腿关节需要大幅折叠/展开，不应被限位惩罚约束，故只保留 ABAD + KENN。
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ABAD_JOINT", ".*_KENN_JOINT"])},
    )

    # 前后腿都不外摆：罚四条腿 ABAD 偏离 0，把腿收回腰部附近而非向两侧张开。
    # joint_deviation_l1 无条件返回 Σ|q_ABAD|（ABAD 默认=0），配合负权重驱动四条腿并拢。
    # 一个正则同时匹配 LF/RF/LB/RB 四条腿的 ABAD 关节，等价于 front+rear 各 -1.0 求和。
    abad_penalty = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ABAD_JOINT")},
    )

    # 前轮不空转：站起后前轮离地，不应无谓转动。罚前轮（LF/RF FOOT）转速。
    # wheel_vel_penalty 里 in_air 门控对前轮（始终离地）恒为 1，running/standing 都退化成「罚前轮转速」；
    # 后轮（驱动轮）不在 asset_cfg 内，不受影响。
    front_wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="[L,R]F_FOOT_LINK"),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names="[L,R]F_FOOT_JOINT"),
        },
    )

    # 前腿收腰：把前腿 HIP/KENN 拉向折叠目标角，让前轮贴着腰部侧面而不是向前伸出。
    # 目标角（HIP≈2.35, KENN≈-1.6）由 URDF 正运动学验证：此时前轮（FOOT_LINK）落在
    # 机体系 (x≈-0.03, y≈0.21, z≈-0.005)（原点 base_link）附近，即贴着腰侧、与腰同高，
    # 且离地（不会触发 front_contact_penalty）。只作用于 LF/RF，不影响支撑用的后腿。
    # 用 gated_joint_pos_target_l2 加「站立进度」门控：机身还平趴时门控≈0，前腿目标角
    # 奖励不生效，避免跟后腿起身抢跑；机身接近竖直后门控→1，才开始把前腿往回收。
    front_hip_tuck = RewTerm(
        func=mdp.gated_joint_pos_target_l2,
        weight=-1.0,
        params={
            "target": 2.35,
            "asset_cfg": SceneEntityCfg("robot", joint_names="[L,R]F_HIP_JOINT"),
        },
    )
    front_kenn_tuck = RewTerm(
        func=mdp.gated_joint_pos_target_l2,
        weight=-1.0,
        params={
            "target": -1.6,
            "asset_cfg": SceneEntityCfg("robot", joint_names="[L,R]F_KENN_JOINT"),
        },
    )

    # 左右腿镜像对称：HIP/KENN（俯仰）左右同号、ABAD（侧摆）左右反号，前后腿都约束。
    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "same_pairs": [
                ["LF_HIP_JOINT", "RF_HIP_JOINT"],
                ["LF_KENN_JOINT", "RF_KENN_JOINT"],
                ["LB_HIP_JOINT", "RB_HIP_JOINT"],
                ["LB_KENN_JOINT", "RB_KENN_JOINT"],
            ],
            "opposite_pairs": [
                ["LF_ABAD_JOINT", "RF_ABAD_JOINT"],
                ["LB_ABAD_JOINT", "RB_ABAD_JOINT"],
            ],
        },
    )

    # ── 继承项移除（None）：站立/关节位形/轮速相关惩罚不适合双轮足 ────────
    stand_still = None
    joint_pos = None
    wheel_vel_penalty = None


# ── 课程配置（前进速度膨胀改监控沿朝向的前向追踪）────────────────────
@configclass
class BipedCurriculumCfg:
    """课程：lin_vel_cmd_levels 改监控沿朝向的 track_lin_vel_heading_exp（不再是机体系）。"""

    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "track_lin_vel_heading_exp"},
    )


# ── 环境（训练）───────────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Biped_Env(RP_wd_Walk_Flat_Env):
    """RP_wd 双轮足（后腿站立）平地行走训练环境（框架同平地，仅命令/奖励不同）。"""

    rewards: BipedRewardsCfg = BipedRewardsCfg()
    commands: BipedCommandsCfg = BipedCommandsCfg()
    curriculum: BipedCurriculumCfg = BipedCurriculumCfg()


# ── 环境（Play）───────────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Biped_Env_Play(RP_wd_Walk_Biped_Env):
    """双轮足 Play 环境（键盘遥控，单环境）。"""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.decimation = 8
        self.episode_length_s = 40
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
