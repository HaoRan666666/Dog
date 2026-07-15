import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

USD_PATH = os.path.dirname(__file__)

LEG_JOINTS = [".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
WHEEL_JOINTS = [".*_FOOT_JOINT"]

RP_wd_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/../assets/RP_wd/usd/RP_wd.usd",
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_ABAD_JOINT": 0.0,
            "[L,R][F,B]_HIP_JOINT": 0.72,
            ".*_KENN_JOINT": -1.41,
            ".*_FOOT_JOINT": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "LEGS": DelayedPDActuatorCfg(
            joint_names_expr=LEG_JOINTS,
            min_delay=0,
            max_delay=2,
            effort_limit={".*": 42},
            effort_limit_sim={".*": 42},
            velocity_limit={".*": 10},
            velocity_limit_sim={".*": 10},
            stiffness={".*": 100.0},
            damping={".*": 3.0},
            armature={".*": 0.02}, #电枢
            friction={".*": 0.02},  
        ),
        "WHEELS": DelayedPDActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,
            min_delay=0,
            max_delay=2,
            effort_limit={".*": 17},
            effort_limit_sim={".*": 17},
            velocity_limit={".*": 42},
            velocity_limit_sim={".*": 42},
            stiffness={".*": 5.0},
            damping={".*": 2},
            armature={".*": 0.01},
            friction={".*": 0.01},
        ),
    },
)
