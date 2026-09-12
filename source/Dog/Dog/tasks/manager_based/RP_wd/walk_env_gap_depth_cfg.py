"""RP_wd 深度感知跨沟：复用深度编码器，独立配置地形、奖励与复位。"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from Dog.assets.terrain.gap_terrain import GAP_TERRAINS_CFG

from . import mdp
from .mdp import gap
from .walk_env_cfg import RP_wd_Walk_Flat_Env, SceneCfg, EventCfg, RewardsCfg, TerminationsCfg
from .walk_env_preceptive_platform_cfg import PlatformDepthSceneCfg, PlatformDepthObservationsCfg


@configclass
class GapDepthSceneCfg(SceneCfg):
    terrain = SceneCfg(num_envs=1, env_spacing=4.0).terrain.replace(
        terrain_type="generator", terrain_generator=GAP_TERRAINS_CFG,
        max_init_terrain_level=0,
    )
    depth_camera = PlatformDepthSceneCfg(num_envs=1, env_spacing=4.0).depth_camera.copy()


@configclass
class GapCommandsCfg:
    """参考 platform 坑地形：只采样前进指令，不采样横移或转向。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(12.0, 12.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.6, 1.0), lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0),
        ),
    )


@configclass
class GapRewardsCfg(RewardsCfg):
    success = RewTerm(func=gap.gap_success_bonus, weight=10.0)
    centerline = RewTerm(func=gap.gap_centerline_penalty, weight=-0.5)
    # 允许腾空与俯仰，避免平地奖励抑制跨沟动作。
    lin_vel_z_l2 = None
    feet_air_time = None
    joint_pos = None
    stand_still = None
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)


@configclass
class GapTerminationsCfg(TerminationsCfg):
    success = DoneTerm(
        func=gap.gap_success,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK")},
    )
    gap_failure = DoneTerm(func=gap.gap_failure)


@configclass
class GapCurriculumCfg:
    terrain_levels = CurrTerm(func=gap.gap_terrain_levels)


@configclass
class RP_wd_Walk_Gap_Depth_Env(RP_wd_Walk_Flat_Env):
    scene: GapDepthSceneCfg = GapDepthSceneCfg(num_envs=256, env_spacing=4.0)
    observations: PlatformDepthObservationsCfg = PlatformDepthObservationsCfg()
    commands: GapCommandsCfg = GapCommandsCfg()
    rewards: GapRewardsCfg = GapRewardsCfg()
    terminations: GapTerminationsCfg = GapTerminationsCfg()
    curriculum: GapCurriculumCfg = GapCurriculumCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.depth_camera.update_period = self.decimation * self.sim.dt
        # +X 前视相机下俯 25°，观察沟沿与地面；沿用 2m 量程及匹配的噪声归一化。
        pitch = math.radians(25.0)
        self.scene.depth_camera.offset.rot = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
        # 空范围表示各轴偏移为零：出生在当前地形原点，使用默认站立高度及 +X 朝向。
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.push_robot = None
        # 先建立跨沟能力，再逐步加大域随机化。
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.base_com.params["com_range"] = {
            "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01),
        }
        self.events.physics_material.params.update(
            static_friction_range=(0.7, 1.2), dynamic_friction_range=(0.6, 1.0),
            restitution_range=(0.0, 0.05),
        )
        generator = self.scene.terrain.terrain_generator
        terrain = generator.sub_terrains["gap"]
        if generator.num_rows != terrain.num_levels or generator.difficulty_range != (0.0, 1.0):
            raise ValueError("Gap terrain rows must match num_levels and difficulty_range must be (0, 1)")
        goal_x = terrain.approach_length + terrain.gap_width_range[1] + terrain.landing_clearance
        if goal_x >= generator.size[0] - terrain.spawn_x - terrain.end_margin:
            raise ValueError("Landing area is too short for the configured gap")
        if not (0 < terrain.lane_half_width < generator.size[1] / 2 - terrain.side_margin):
            raise ValueError("The allowed lane must remain inside the open gap width")


@configclass
class RP_wd_Walk_Gap_Depth_Env_Play(RP_wd_Walk_Gap_Depth_Env):
    """自动前进评估，保留成功/失败终止，关闭课程与随机深度增强。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        self.curriculum = None
        self.observations.policy.enable_corruption = False
        self.scene.depth_camera.noise_pipeline = {
            k: v for k, v in self.scene.depth_camera.noise_pipeline.items()
            if k in ("gaussian_blur", "depth_normalization")
        }
