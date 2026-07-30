import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

USD_PATH = os.path.dirname(__file__)

# ── 关节分组 ──────────────────────────────────────────────────────────
BODY_JOINTS = ["body_yaw_joint"]                                          # 腰部 yaw (1 DOF)
ARM_JOINTS = ["shoulder_.*", "elbow_.*"]                               # 双臂 (8 DOF)
LEG_JOINTS = ["thigh_.*", "knee_.*", "foot_.*"]                       # 双腿 (12 DOF)

RPMini_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/../assets/RPMini/usd/RPMini.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=2,
            fix_root_link=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01, rest_offset=0.0,
            torsional_patch_radius=0.01, min_torsional_patch_radius=0.003,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "shoulder_left_pitch_joint": 0.1,
            "shoulder_right_pitch_joint": 0.1,
            "shoulder_left_roll_joint": 0.3,
            "shoulder_right_roll_joint": -0.3,
            "elbow_left_pitch_joint": 0.8,
            "elbow_right_pitch_joint": 0.8,
            "thigh_left_pitch_joint": -0.2,
            "thigh_right_pitch_joint": -0.2,
            "knee_left_pitch_joint": 0.35,
            "knee_right_pitch_joint": 0.35,
            "foot_left_pitch_joint": -0.18,
            "foot_right_pitch_joint": -0.18,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                "thigh_.*_roll_joint",
                "thigh_.*_pitch_joint",
                "knee_.*_yaw_joint",
                "knee_.*_pitch_joint",
                "body_yaw_joint",
            ],
            effort_limit_sim=100,
            velocity_limit_sim=50,
            stiffness={
                "thigh_.*_roll_joint": 100.0,
                "thigh_.*_pitch_joint": 100.0,
                "knee_.*_yaw_joint": 150.0,
                "knee_.*_pitch_joint": 150.0,
                "body_yaw_joint": 150.0,
            },
            damping={
                "thigh_.*_roll_joint": 3.3,
                "thigh_.*_pitch_joint": 3.3,
                "knee_.*_yaw_joint": 5.0,
                "knee_.*_pitch_joint": 5.0,
                "body_yaw_joint": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=["foot_.*_pitch_joint", "foot_.*_roll_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=50,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                "shoulder_.*_pitch_joint",
                "shoulder_.*_roll_joint",
            ],
            effort_limit_sim=100,
            velocity_limit_sim=50,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                "elbow_.*_pitch_joint",
                "elbow_.*_yaw_joint",
            ],
            stiffness={
                "elbow_.*_pitch_joint": 30.0,
                "elbow_.*_yaw_joint": 20.0,
            },
            damping={
                "elbow_.*_pitch_joint": 1.5,
                "elbow_.*_yaw_joint": 1.0,
            },
            effort_limit_sim=100,
            velocity_limit_sim=50,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)
