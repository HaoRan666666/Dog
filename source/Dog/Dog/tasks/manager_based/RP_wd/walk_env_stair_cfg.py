"""RP_wd 深度相机上台阶任务配置。

任务描述：
    在 ``walk_env_terrain_cfg``（地形行走）基础上，参考 parkour 的深度相机 pipeline，
    新增一台前视下倾的深度相机，把「带噪声 + 历史缓存的延迟多帧深度序列」作为观测
    同时喂给 policy 与 critic，让策略学会「用深度相机看台阶」爬升。

与 ``walk_env_terrain_cfg`` 的区别：
    - 场景：多一台 ``depth_camera``（NoisyGroupedRayCasterCamera，挂 base_link，前视下倾），
      参考 parkour 深度相机配置：加噪声 pipeline + 37 帧历史缓存 + 自身连杆 raycast 目标。
    - 观测：policy 与 critic 的深度图改为 ``delayed_visualizable_image``（延迟多帧深度序列），
      观测组从「扁平向量」(concatenate_terms=True) 改为「分项 dict」(concatenate_terms=False)。
    - 命令 / 奖励 / 终止 / 课程 / 复位 全部继承地形配置。

注意：观测改为 dict + 图像后，需要 vision-capable 的策略网络（RSL-RL 默认 MLP 只能消费
扁平向量，不能直接消费 dict + (N,8,H,W) 图像）。策略网络需自行拼接标量项并加 CNN/Transformer
处理深度序列。
"""

from isaaclab.devices import DevicesCfg
from isaaclab.devices.keyboard import Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from Dog.robots.RP_wd import RP_wd_LINKS
from Dog.sensor.grouped_ray_caster import get_link_prim_targets
from Dog.sensor.noisy_camera import NoisyGroupedRayCasterCameraCfg
from Dog.utils.noise import (
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    PerlinNoiseCfg,
    PixelFailureNoiseCfg,
    RandomConvNoiseCfg,
    ScaleRandomizationNoiseCfg,
    StereoFusionNoiseCfg,
)

from . import mdp
from .walk_env_cfg import LEG_JOINTS
from .walk_env_terrain_cfg import (
    RP_wd_Walk_Terrain_Env,
    TerrainSceneCfg,
    TerrainObservationsCfg,
)


@configclass
class StairSceneCfg(TerrainSceneCfg):
    """台阶场景：继承地形场景，新增前视下倾的深度相机。"""

    # 深度相机：挂 base_link，前视下倾 ~45°，看前方台阶。
    # 相机光轴为 +X（pinhole_camera_pattern 输出 x forward / y left / z up），
    # 故用 convention="world"（forward=+X, up=+Z）。rot 绕 +Y 转 +45° 即「低头」。
    # 姿态为占位值，需在 viewer 里验证视角后再微调 pos/rot。
    depth_camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(0.29, 0.0, 0.07),
            rot=(0.9238795, 0.0, 0.3826834, 0.0),  # 绕 +Y 转 +45°（低头）
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=2.0,
        min_distance=0.1,
        update_period=0.02,  # 每控制步 (decimation*sim.dt = 0.02s) 更新一帧
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            width=64,
            height=48,
        ),
        # 地面 + 机器人自身连杆（让深度图能「看到」自己的腿，避免盲区/自遮挡信息丢失）
        mesh_prim_paths=["/World/ground", *get_link_prim_targets(RP_wd_LINKS)],
        debug_vis=False,
        # 仿真→真实深度退化（噪声 pipeline，作用于归一化前的原始 metric 深度）
        noise_pipeline={
            # --- 保守增强（作用于原始 metric 深度）---
            "scale_randomization": ScaleRandomizationNoiseCfg(
                apply_probability=0.5,
                scale_min=0.97,
                scale_max=1.03,
            ),
            "stereo_fusion": StereoFusionNoiseCfg(
                apply_probability=0.4,
                disparity_grad_threshold=0.10,
                texture_var_threshold=3e-4,
                hole_probability=0.02,
                hole_kernel_size=1,
                hole_value=2.0,  # 无数据按最大距离 2.0m
            ),
            "random_conv": RandomConvNoiseCfg(
                apply_probability=0.3,
                kernel_std=0.05,
                center_weight=1.0,
            ),
            "perlin_noise": PerlinNoiseCfg(
                apply_probability=0.5,
                octaves=3,
                base_frequency=8.0,
                lacunarity=2.0,
                persistence=0.5,
                amplitude=1.0,
                noise_std=0.01,
            ),
            "pixel_failures": PixelFailureNoiseCfg(
                apply_probability=0.5,
                dead_pixel_prob=5e-4,
                saturated_pixel_prob=5e-4,
                dead_value=0.0,
                saturated_value=2.0,  # 饱和 = 最大距离（归一化前）
            ),
            # --- 固定预处理（保持最后）---
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.0),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},
    )


@configclass
class StairObservationsCfg(TerrainObservationsCfg):
    """台阶观测：标量项保留 history_length=3，深度图改为延迟多帧序列；分项 dict 输出。

    ``concatenate_terms=False``：观测以 dict 返回（标量项 + 深度图分开），
    由 vision-capable 策略网络分别处理。标量项各自叠加 3 步历史并展平；
    深度图 ``delayed_visualizable_image`` 内部已采样 8 帧延迟序列，不再叠加 manager 历史。
    """

    @configclass
    class PolicyCfg(TerrainObservationsCfg.PolicyCfg):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=3, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=3, flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"},
            history_length=3, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            history_length=3, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=3, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=3, flatten_history_dim=True)
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            super().__post_init__()
            # 深度图是多帧图像张量，不能和标量拼接成扁平向量，改分项 dict 输出。
            self.concatenate_terms = False
            # 关闭 group 级历史覆盖：历史长度改由每项自控（标量=3，深度图=0）。
            self.history_length = None

    @configclass
    class CriticCfg(TerrainObservationsCfg.CriticCfg):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=3, flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=3, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=3, flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"},
            history_length=3, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            history_length=3, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=3, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=3, flatten_history_dim=True)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            history_length=3, flatten_history_dim=True,
        )
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = False
            self.history_length = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RP_wd_Walk_Stair_Env(RP_wd_Walk_Terrain_Env):
    """RP_wd 深度相机上台阶训练环境（框架同地形，仅场景加相机、观测加深度图）。"""

    scene: StairSceneCfg = StairSceneCfg(num_envs=2048, env_spacing=4.0)
    observations: StairObservationsCfg = StairObservationsCfg()


@configclass
class RP_wd_Walk_Stair_Env_Play(RP_wd_Walk_Stair_Env):
    """台阶 Play 环境（键盘遥控，单环境，开深度相机 debug_vis 验证占位姿态）。"""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.decimation = 8
        self.episode_length_s = 40
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.observations.policy.enable_corruption = False

        # 保留全部地形（上/下台阶 + 平地），便于遥控验证深度相机视角
        self.scene.terrain.max_init_terrain_level = 15
        # 开相机调试可视化：显示相机 frame 与射线命中点（红球），用于验证占位姿态
        self.scene.depth_camera.debug_vis = True

        self.teleop_devices = DevicesCfg({
            "keyboard": Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=1.0,
                omega_z_sensitivity=2.0,
            ),
        })

        self.curriculum = None
        self.terminations.base_contact = None
