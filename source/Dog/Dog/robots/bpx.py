import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

USD_PATH = os.path.dirname(__file__)

BPX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_PATH}/../assets/bpx/urdf/bpx.usd",
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
            ".*_hip_roll_joint": 0.0,
            "f[l,r]_hip_pitch_joint": 0.8,
            "h[l,r]_hip_pitch_joint": 1.0,
            ".*_knee_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "JOINTS": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
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
