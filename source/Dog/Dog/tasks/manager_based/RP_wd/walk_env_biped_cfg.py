"""双轮足（只用两条后腿）平地行走训练环境配置。

任务描述：
    RP_wd 从与平地完全相同的四腿站立姿态出发，仅通过奖励学习用两条后腿（LB/RB）
    + 后轮支撑并行走，前腿（LF/RF）抬起离地，形成「双轮足」倒立摆平衡任务。

与 ``walk_env_cfg``（四轮足平地走）的区别（只改奖励）：
    - 动作 / 观测 / 指令 / 事件 / 终止 / 课程 / 初始姿态 全部与平地一致。
    - 仅多一条「前腿 + 前轮触地惩罚」，逼策略把前腿抬起来、后腿站立。
"""

from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .walk_env_cfg import (
    RP_wd_Walk_Flat_Env,
    RewardsCfg as FlatRewardsCfg,
)


# ── 奖励配置（平地奖励 + 前腿触地惩罚）───────────────────────────────
@configclass
class BipedRewardsCfg(FlatRewardsCfg):
    """奖励：完全继承平地，额外加一条「前腿/前轮触地惩罚」驱动后腿站立。"""

    # 前腿任意连杆（ABAD/HIP/KENN/FOOT）触地 → 重罚。
    # 四腿站立时前轮持续着地，会一直吃到这条惩罚；策略要消除它只能把前腿抬起来，
    # 用两条后腿 + 后轮支撑行走。权重与 base_contact_penalty 相当，可再加大。
    front_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="[L,R]F_.*_LINK"),
            "threshold": 1.0,
        },
    )


# ── 环境（训练）───────────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Biped_Env(RP_wd_Walk_Flat_Env):
    """RP_wd 双轮足（后腿站立）平地行走训练环境（框架同平地，仅奖励不同）。"""

    rewards: BipedRewardsCfg = BipedRewardsCfg()


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
