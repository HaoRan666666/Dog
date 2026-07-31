import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# 机器人配置
##
from Dog.robots.RPMini import RPMini_CFG

# ── 关节分组 ──────────────────────────────────────────────────────────
BODY_JOINTS = ["body_yaw_joint"]
ARM_JOINTS = [
    "shoulder_left_pitch_joint", "shoulder_left_roll_joint",
    "elbow_left_yaw_joint", "elbow_left_pitch_joint",
    "shoulder_right_pitch_joint", "shoulder_right_roll_joint",
    "elbow_right_yaw_joint", "elbow_right_pitch_joint",
]
LEG_JOINTS = [
    "thigh_left_pitch_joint", "thigh_left_roll_joint",
    "thigh_right_pitch_joint", "thigh_right_roll_joint",
    "knee_left_yaw_joint", "knee_left_pitch_joint",
    "knee_right_yaw_joint", "knee_right_pitch_joint",
]
FOOT_JOINTS = [
    "foot_left_pitch_joint", "foot_left_roll_joint",
    "foot_right_pitch_joint", "foot_right_roll_joint",
]

# ── 场景配置 ──────────────────────────────────────────────────────────
@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot = RPMini_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 左右脚掌高度扫描器
    left_feet_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/foot_left_roll_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.12, 0.04]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    right_feet_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/foot_right_roll_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.12, 0.04]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ── 指令配置 ──────────────────────────────────────────────────────────
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 3.0), lin_vel_y=(-3.0, 3.0), ang_vel_z=(-1.5, 1.5)
        ),
    )


# ── 观测配置 ──────────────────────────────────────────────────────────
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ── 动作配置 ──────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, preserve_order=True
    )


# ── 奖励配置 ──────────────────────────────────────────────────────────
@configclass
class RewardsCfg:
    # -- 任务跟踪
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.8, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.7, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- 运动惩罚
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.03)

    # -- 平滑惩罚
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    energy = RewTerm(func=mdp.energy, weight=-2e-4)

    # -- 接触惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "thigh_.*", "shoulder_.*", "elbow_.*"]),
            "threshold": 1.0,
        },
    )

    # -- 姿态约束
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.5)

    # -- 关节约束
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )

    # ── 关节偏离惩罚：四类不同权重 ─────────────────────────────────────
    joint_deviation_hip = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "thigh_left_roll_joint", "thigh_right_roll_joint",
            ]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "body_yaw_joint",
                "shoulder_left_roll_joint", "shoulder_right_roll_joint",
                "elbow_left_yaw_joint", "elbow_right_yaw_joint",
                "elbow_left_pitch_joint", "elbow_right_pitch_joint",
            ]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.06,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "shoulder_left_pitch_joint", "shoulder_right_pitch_joint",
            ]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "thigh_left_pitch_joint", "thigh_right_pitch_joint",
                "knee_left_yaw_joint", "knee_right_yaw_joint",
                "knee_left_pitch_joint", "knee_right_pitch_joint",
                "foot_left_pitch_joint", "foot_right_pitch_joint",
                "foot_left_roll_joint", "foot_right_roll_joint",
            ]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-1,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*roll.*"), "threshold": 0.3},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*roll.*"),
        },
    )

    # feet_force = RewTerm(
    #     func=mdp.body_force,
    #     weight=-3e-3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*roll.*"),
    #         "threshold": 500,
    #         "max_reward": 400,
    #     },
    # )

    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*foot.*roll.*"])},
    )

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "foot_height":0.04,"threshold":0.02})

    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*roll.*"), "min": 0.14, "max": 0.4},
    )

    # knee_distance = RewTerm(
    #     func=mdp.body_distance_y,
    #     weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*knee.*pitch.*"]), "min": 0.16, "max": 0.3},
    # )

# ── 终止条件 ──────────────────────────────────────────────────────────
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link", "thigh_.*", "shoulder_.*", "elbow_.*"],
            ),
            "threshold": 1.0,
        },
    )
        # ── 基座高度过低 ────────────────────────────────────────────────────
    # 基座离地高度低于 0.1m → 终止
    # 正常站立时基座约 0.5m, 0.1m 意味着蹲下/摔倒
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.1},
    )

    # ── 躯干过度倾斜 ────────────────────────────────────────────────────
    # 躯干倾斜超过 60° (相对垂直方向) → 终止
    # limit_angle=60°: 超过这个角度意味着机器人已经失控
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )



# ── 课程配置 ──────────────────────────────────────────────────────────
@configclass
class CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


# ── 域随机化 ──────────────────────────────────────────────────────────
@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.2, 1.2),
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
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    leg_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="thigh_.*|knee_.*|foot_.*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.03, 0.03)},
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (0.0, 0.0)},
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-1.0, 1.0)}},
    )


# ── 平地步态环境 ──────────────────────────────────────────────────────
@configclass
class RPMini_Walk_Flat_Env(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 8
        self.episode_length_s = 12
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
