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
            "body_yaw_joint": 0.0,
            "shoulder_left_pitch_joint": 0.0,
            "shoulder_right_pitch_joint": 0.0,
            "shoulder_left_roll_joint": 0.5,
            "shoulder_right_roll_joint": -0.5,
            "elbow_left_yaw_joint": 0.0,
            "elbow_right_yaw_joint": 0.0,
            "elbow_left_pitch_joint": 0.0,
            "elbow_right_pitch_joint": 0.0,
            "thigh_right_roll_joint": -0.3,
            "thigh_left_roll_joint": -0.3,
            "knee_right_pitch_joint": 0.6,
            "knee_left_pitch_joint": 0.6,
            "foot_right_pitch_joint": -0.3,
            "foot_left_pitch_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "BODY": DelayedPDActuatorCfg(
            joint_names_expr=BODY_JOINTS,
            min_delay=0, 
            max_delay=0,
            effort_limit=25, 
            effort_limit_sim=25,
            velocity_limit=13.61, 
            velocity_limit_sim=13.61,
            stiffness=100.0, 
            damping=3.0,
            armature=0.02, 
            friction=0.02, 
            dynamic_friction=0.01,
        ),
        "ARMS": DelayedPDActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            min_delay=0, 
            max_delay=0,
            effort_limit=25, 
            effort_limit_sim=25,
            velocity_limit=13.61, 
            velocity_limit_sim=13.61,
            stiffness=80.0, 
            damping=3.0,
            armature=0.01, 
            friction=0.01, 
            dynamic_friction=0.005,
        ),
        "LEGS": DelayedPDActuatorCfg(
            joint_names_expr=LEG_JOINTS,
            min_delay=0, max_delay=0,
            effort_limit=25,
            effort_limit_sim=25,
            velocity_limit=13.61, 
            velocity_limit_sim=13.61,
            stiffness=100.0, 
            damping=3.0,
            armature=0.02, 
            friction=0.02, 
            dynamic_friction=0.01,
        ),
    },
)
