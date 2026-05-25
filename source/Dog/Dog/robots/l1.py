import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

USD_PATH = os.path.dirname(__file__)

L1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/../assets/L1/urdf/ZSL-1/ZSL-1.usd",
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
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            ".*_ABAD_JOINT": 0.0,
            "F[L,R]_HIP_JOINT": 0.8,
            "R[L,R]_HIP_JOINT": 1.0,
            ".*_KNEE_JOINT": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "JOINTS": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            min_delay=0,
            max_delay=2,
            effort_limit={".*": 50},
            effort_limit_sim={".*": 30},
            velocity_limit={".*": 26},
            velocity_limit_sim={".*": 25},
            stiffness={".*": 20.0},
            damping={".*": 1.0},
            armature={".*": 0.01},
            friction={".*": 0.2},
        ),
    },
)
