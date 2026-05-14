import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg,DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os 

USD_PATH= os.path.dirname(__file__)
print(f"USD_PATH: {USD_PATH}")
# ── L1 四足机器人配置 ───────────────────────────────────────────────
# 关节结构 (每条腿 3 个驱动关节 + 1 个固定足端):
#   ABAD_JOINT (髋侧摆) → HIP_JOINT (髋俯仰) → KNEE_JOINT (膝俯仰) → FOOT_JOINT (固定)
# 腿命名: FL=前左, FR=前右, RL=后左, RR=后右
L1_CFG = ArticulationCfg(
    # ── 加载 USD 模型 ──────────────────────────────────────────────
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/../assets/L1/urdf/ZSL-1/ZSL-1.usd",
        # 开启接触传感器，用于足端力感知
        activate_contact_sensors=True,
        # 刚体属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,          # 受重力影响
            retain_accelerations=False,     # 不保留上一帧加速度
            linear_damping=0.0,             # 无额外线速度阻尼
            angular_damping=0.0,            # 无额外角速度阻尼
            max_linear_velocity=1000.0,     # 线速度上限 (m/s)
            max_angular_velocity=1000.0,    # 角速度上限 (rad/s)
            max_depenetration_velocity=1.0, # 去穿透速度 (m/s)
        ),
        # 关节链根属性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,              # 禁用自碰撞
            solver_position_iteration_count=4,          # 位置求解迭代次数
            solver_velocity_iteration_count=0,          # 速度求解迭代次数 (与位置解耦)
            fix_root_link=False,
        ),
    ),
    # ── 初始状态 ───────────────────────────────────────────────────
    init_state=ArticulationCfg.InitialStateCfg(
        # 机身初始位置: 离地 0.4m 站立
        pos=(0.0, 0.0, 0.3),
        # 初始关节角度 (站立姿态)
        joint_pos={
            # ABAD 关节: 全部归零 (腿在竖直平面内)
            ".*_ABAD_JOINT": 0.0,
            # HIP 关节: 前腿微前伸，后腿微后伸
            "F[L,R]_HIP_JOINT": 0.8,
            "R[L,R]_HIP_JOINT": 1.0,
            # KNEE 关节: 弯曲使重心降低 (KNEE 限位为负值, -1.5 在限位 [-2.723, -0.602] 范围内)
            ".*_KNEE_JOINT": -1.5,
        },
        # 初始关节速度: 全部静止
        joint_vel={".*": 0.0},
    ),
    # ── 关节软限位 ─────────────────────────────────────────────────
    # 关节指令不会超过 [下限 * factor, 上限 * factor] 的范围
    soft_joint_pos_limit_factor=0.9,
    # ── 执行器配置 ─────────────────────────────────────────────────
    actuators={
        "JOINTS": DelayedPDActuatorCfg(
          joint_names_expr=[".*_ABAD_JOINT",".*_HIP_JOINT",".*_KNEE_JOINT"],
          min_delay=0,
          max_delay=2,
            effort_limit={
                ".*": 50
                }, 
            effort_limit_sim={
                ".*": 30
            }, 
            velocity_limit={
                ".*": 26
            },
            velocity_limit_sim={
                ".*": 25
            },
            stiffness={
                ".*": 20.0
            },
            damping={
                ".*": 1.0
            },
            armature={
                ".*": 0.01
            },
            friction={
                ".*": 0.2
            },
        ),
    },
)
