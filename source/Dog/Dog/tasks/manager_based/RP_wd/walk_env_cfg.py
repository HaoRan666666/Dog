
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from . import mdp
##
# 预定义配置导入
##
from Dog.robots.RP_wd import RP_wd_CFG 
from Dog.assets.terrain.step_terrain import STEP_TERRAINS_CFG

LEG_JOINTS = [".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
WHEEL_JOINTS = [".*_FOOT_JOINT"]

# ── 场景配置（平地步态用）──────────────────────────────────────────
@configclass
class SceneCfg(InteractiveSceneCfg):
   #地形配置
   terrain = TerrainImporterCfg(
       prim_path  ="/World/ground",#表示 USD 场景中 terrain 根 prim 的绝对路径。
       terrain_type="plane",
       terrain_generator=STEP_TERRAINS_CFG,
       max_init_terrain_level=None,
       collision_group=-1,
       physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0, #这两个是摩擦力参数，
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,#关闭调试可视化,
   )
   robot = RP_wd_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
   height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.4)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
   contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
   sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,#太阳级亮度
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
   

@configclass
class CommandsCfg: #MDP（马尔可夫决策过程）指令生成器的配置类
    """Command specifications for the MDP."""

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
            lin_vel_x=(-2.5, 2.5), lin_vel_y=(-2.0, 2.0), ang_vel_z=(-1.5, 1.5)
        ),
    )

@configclass 
class ObservationsCfg:
    @configclass 
    class PolicyCfg(ObsGroup):
        # base_lin_vel=ObsTerm(func=mdp.base_lin_vel,noise=Unoise(n_min=-0.1,n_max=0.1))
        base_ang_vel=ObsTerm(func=mdp.base_ang_vel,noise=Unoise(n_min=-0.2,n_max=0.2))
        projected_gravity=ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True


    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel=ObsTerm(func=mdp.base_lin_vel,noise=Unoise(n_min=-0.1,n_max=0.1))
        base_ang_vel=ObsTerm(func=mdp.base_ang_vel,noise=Unoise(n_min=-0.2,n_max=0.2))
        projected_gravity=ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
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
    

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
#策略网络输出的 action 是什么格式，以及如何把神经网络的输出映射到机器人关节上。
    joint_pos_abad = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*_ABAD_JOINT"], scale=0.125, use_default_offset=True,preserve_order=True)
    joint_pos_legs = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*_HIP_JOINT", ".*_KENN_JOINT"], scale=0.25, use_default_offset=True,preserve_order=True)
    joint_pos_wheels = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*_FOOT_JOINT"], scale=5.0, use_default_offset=False,preserve_order=True)
#use_default_offset=True  让机器人动作中心从"0"变成"默认站立姿态"。
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.8, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    #抑制竖直速度过大
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    # #横滚和俯仰角速度
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # #电机力矩
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
#   #关节加速度
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
#   #动作变化
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    energy = RewTerm(func=mdp.energy, weight=-2e-4,
                     params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK"),
            "command_name": "base_velocity",
            "threshold": 0.8,
        },
    )
    
    # base_link 触地独立惩罚（后倒/侧翻直接标志），权重更高
    base_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), "threshold": 1.0},
    )
    # 大腿/小腿触地惩罚（蹭台阶等，相对可容忍）
    leg_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*HIP_LINK", ".*KENN_LINK"]), "threshold": 1.0},
    )
#     # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-20.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
    )

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )


    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-1,
        params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
              },
    ) 

    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK"),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*FOOT_JOINT"),
        },
    )


#     trotting_rew= RewTerm(
#         func= mdp.GaitReward,
#         weight= 0.2,
#         params={
#             "std": math.sqrt(0.5),
#             "command_name": "base_velocity",
#             "max_err": 0.2,
#             "velocity_threshold": 0.5,
#             "command_threshold": 0.1,
#             "synced_feet_pair_names": [
#             [".*LF.*FOOT.*", ".*RB.*FOOT.*"],
#             [".*RF.*FOOT.*", ".*LB.*FOOT.*"]],
#             "asset_cfg": SceneEntityCfg("robot"),
#             "sensor_cfg": SceneEntityCfg("contact_forces"),
#         }
#     )

#     feet_air_time_variance = RewTerm(
#     func=mdp.feet_air_time_variance_penalty,
#     weight=-0.1,
#     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK")},
#     )


#     diagonal_joint_mirror = RewTerm(
#     func=mdp.joint_mirror,
#     weight=-1,
#     params={
#         "asset_cfg": SceneEntityCfg("robot"),
#         "mirror_joints": [
#             ["LF_HIP_JOINT", "RB_HIP_JOINT"],
#             ["LF_KENN_JOINT", "RB_KENN_JOINT"],

#             ["RF_HIP_JOINT", "LB_HIP_JOINT"],
#             ["RF_KENN_JOINT", "LB_KENN_JOINT"],
#         ],
#     },
# )
    
#     foot_slide=RewTerm(
#         func=mdp.feet_slide,
#         weight=-0.3,
#            params={
#             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK"),
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT_LINK"),
#         },
#     )

#     sound_suppression = RewTerm(
#         func=mdp.sound_suppression_acc_per_foot,
#         weight=-0.002,
#         params={
#             "sensor_cfg": SceneEntityCfg(
#                 "contact_forces",
#                 body_names=".*FOOT_LINK",
#             ),
#         },
#     )




    # hipx_joint_pos_penalty = RewTerm(
    #     func=mdp.joint_pos_penalty,
    #     weight=-0.4,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_ABAD_JOINT"),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.5,
    #         "command_threshold": 0.1,
    #     },
    # )

    # hipy_joint_pos_penalty = RewTerm(
    #     func=mdp.joint_pos_penalty,
    #     weight=-0.2,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_HIP_JOINT"),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.5,
    #         "command_threshold": 0.1,
    #     },
    # )

    # KENN_JOINT_pos_penalty = RewTerm(
    #     func=mdp.joint_pos_penalty,
    #     weight=-2,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_KENN_JOINT"),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.5,
    #         "command_threshold": 0.1,
    #     },
    # )




    # feet_contact_without_cmd = RewTerm(
    #     func=mdp.feet_contact_without_cmd,
    #     weight=0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK"),
    #         "command_name": "base_velocity",
    #     },
    # )


#     feet_distance_xy = RewTerm(


#     func=mdp.feet_distance_xy_exp,
#     weight=0.4,
#     params={
#         "asset_cfg": SceneEntityCfg(
#             "robot",
#             body_names=[
#                 ".*LF.*FOOT.*",
#                 ".*RF.*FOOT.*",
#                 ".*LB.*FOOT.*",
#                 ".*RB.*FOOT.*",
#             ],
#             preserve_order=True,
#         ),
#         "stance_width": 0.1,
#         "stance_length": 0.2,
#         "std": 0.10,
#     },
# )

    # diagonal_action_sync = RewTerm(
    # func=mdp.action_sync,
    # weight=-0.02,
    # params={
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "joint_groups": [
    #         ["LF_ABAD_JOINT", "RB_ABAD_JOINT"],
    #         ["LF_HIP_JOINT", "RB_HIP_JOINT"],
    #         ["LF_KENN_JOINT", "RB_KENN_JOINT"],

    #         ["RF_ABAD_JOINT", "LB_ABAD_JOINT"],
    #         ["RF_HIP_JOINT", "LB_HIP_JOINT"],
    #         ["RF_KENN_JOINT", "LB_KENN_JOINT"],
    #     ],
    # },
    # )
    
@configclass
class TerminationsCfg:

    """Termination terms for the MDP."""
#定义 episode 结束的条件
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm( #base接触力大于阈值的时候停止episode
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
 
    # thigh_contact = DoneTerm( #大腿触地（侧翻检测）
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*HIP_LINK"), "threshold": 10.0},
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class EventCfg:#定义训练过程中的一些事件
     #分为三种  ：startup、reset、interval
    """Configuration for events."""

    # startup
   # startup 类事件（只在环境初始化时运行一次）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, #材料随机化
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 2.0),
            "dynamic_friction_range": (0.15, 1.5),
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
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # ── 腿部质量缩放 ────────────────────────────────────────────────────
    # 随机缩放左右腿各连杆的质量 (80%~120%)
    # 模拟: 加工公差、材料密度不均匀
    leg_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="[L,R][F,B]_.*_LINK"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (-0.05, 0.05)},
        },
    )

        # ── 腿部执行器增益随机化 ─────────────────────────────────────────────
    # 随机缩放腿部 PD 控制器刚度和阻尼 (80%~120%)
    scale_leg_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    # ── 轮子执行器阻尼随机化 ─────────────────────────────────────────────
    # 轮子 stiffness=0，只随机阻尼 (60%~140%)，刚度不动
    scale_wheel_actuator_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
            "stiffness_distribution_params": (1.0, 1.0),   # stiffness=0，不变
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    # ── 腿部关节参数随机化 ────────────────────────────────────────────────
    # 随机缩放腿部关节电枢惯量 (60%~140%)
    scale_leg_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    # ── 轮子关节参数随机化 ────────────────────────────────────────────────
    # 随机缩放轮子关节电枢惯量 (60%~140%)
    scale_wheel_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )


    # reset
    #reset 类事件（每次 reset / episode 开始触发）
    
    #这里并没有设置外部力（设置为0）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    #reset 时随机化机器人在 xy yaw 的位姿 + 速度。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    #interval 类事件（训练中周期性触发）
    #训练过程中，每 10~15 秒随机在 x,y 方向给一个速度，从而实现推力扰动。
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0),"yaw": (-2.0, 2.0)}},
    )

# ── 平地步态环境 ──────────────────────────────────────────────────
@configclass
class RP_wd_Walk_Flat_Env(ManagerBasedRLEnvCfg):
    """RCdog 四足机器人平地行走环境。"""
    # 场景设置：4096 个并行环境，每个间隔 4m
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=4.0)
    # MDP 设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # # MDP 设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg=CommandsCfg()
    curriculum: CurriculumCfg=CurriculumCfg()

    def __post_init__(self) -> None:
        """初始化后回调：设置仿真步长、回合长度等参数。"""
        # 通用设置
        self.decimation = 8                       # 每 8 步物理仿真执行一次控制
        self.episode_length_s = 12                # 每个 episode 最大时长 (s)
        # 视角设置
        self.viewer.eye = (8.0, 0.0, 5.0)        # 相机默认位置 (x, y, z)
        # 仿真设置
        self.sim.dt = 0.0025                      # 物理 400Hz，控制 400/8=50Hz
        self.sim.render_interval = self.decimation # 渲染间隔


@configclass
class RP_wd_Walk_Flat_Env_Play(RP_wd_Walk_Flat_Env):

    def __post_init__(self) -> None:

        self.scene.num_envs=1
        self.scene.env_spacing=2.5

        self.decimation = 8
        self.episode_length_s = 40
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.observations.policy.enable_corruption = False

        # 键盘控制: WASD 控制线速度, Q/E 控制转向
        self.teleop_devices = DevicesCfg({
            "keyboard": Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=1.0,
                omega_z_sensitivity=2.0,
            ),
        })

        # Play 模式不需要 curriculum
        self.curriculum = None


