"""带「碰撞分组」的光线投射相机（Grouped Ray-Caster Camera）。

这个类把「针孔相机」和「分组光线投射」两套能力合并成一个传感器：

1. **相机视角**（来自 ``RayCasterCamera``）：根据针孔相机内参（焦距/光圈/分辨率），
   给每个像素生成一条从相机原点出发的光线，并把求交结果转成相机系里的深度。
2. **分组求交**（来自 ``GroupedRayCaster``）：对多个 mesh（地面 + 机器人各连杆）做
   GPU 并行光线-三角面求交，同时给每条光线 / 每个 mesh 打「碰撞组 id」（= 环境编号），
   保证每个环境的光线只打到自己环境里的 mesh，避免 2000+ 并行环境之间互相串扰。

因此它只能输出「几何量」（深度、法向、命中点），不能输出 RGB / 语义分割等纹理量，
见 ``UNSUPPORTED_TYPES``。

数据流（每步 ``_update_buffers_impl``）：
    base_link 位姿 ⊕ offset → 传感器世界位姿
    → 把相机系光线变换到世界系
    → Warp kernel 求交（分组）
    → 命中向量逆旋回相机系取 +X 分量 = ``distance_to_image_plane``（像平面深度）。
"""

from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import isaacsim.core.utils.stage as stage_utils
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import CameraData
from isaaclab.sensors.ray_caster import RayCasterCamera
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view

from Dog.utils.warp.raycast import raycast_mesh_grouped

from . import GroupedRayCaster

if TYPE_CHECKING:
    from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg

# import logger
logger = logging.getLogger(__name__)


class GroupedRayCasterCamera(RayCasterCamera, GroupedRayCaster):
    """分组光线投射相机传感器。

    多重继承：``RayCasterCamera`` 负责「相机」部分（内参、逐像素光线、输出缓冲），
    ``GroupedRayCaster`` 负责「分组 mesh 求交」部分（碰撞组、Warp kernel）。
    """

    cfg: GroupedRayCasterCameraCfg

    """The configuration parameters."""
    # 本相机不支持的数据类型：凡是依赖纹理/材质/渲染的（RGB、各类分割、包围盒等）都禁掉，
    # 因为光线投射只能拿到几何量。init 时若配置了这些类型会报错。
    UNSUPPORTED_TYPES: ClassVar[set[str]] = {
        "rgb",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "semantic_segmentation",
        "semantic_segmentation_fast",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: GroupedRayCasterCameraCfg):
        """初始化相机。

        Args:
            cfg: 相机的配置参数（含针孔 pattern、offset 位姿、data_types 等）。

        Raises:
            ValueError: 若 ``cfg.data_types`` 里包含本相机不支持的纹理类数据类型。
        """
        # 检查请求的数据类型是否都在支持范围内
        self._check_supported_data_types(cfg)
        # 初始化基类（按 MRO 依次调用，完成 warp mesh 缓存、内参、光线、输出缓冲等的构建）
        super().__init__(cfg)
        # 创建存放输出数据的容器（rgb/depth/normals 等都会写进这里）
        self._data = CameraData()

    def __str__(self) -> str:
        """返回：描述本实例的一串信息。"""
        return (
            f"Grouped-Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Implementations.
    """

    def _initialize_warp_meshes(self):
        # 复用 GroupedRayCaster 的 mesh 初始化：把每个 mesh 的 warp 表示与「碰撞组 id」建立映射。
        # （RayCasterCamera 本身不含分组逻辑，这里显式指定走 GroupedRayCaster 的版本。）
        GroupedRayCaster._initialize_warp_meshes(self)

    def _initialize_rays_impl(self):
        # 所有环境索引的占位张量，后续多处用它表示「全体环境」。
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # 每个环境的帧计数（供延迟/噪声等按帧调度的逻辑使用）。
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        # 分配各类输出缓冲（rgb/depth/normals 等，由基类实现）。
        self._create_buffers()
        # 由针孔相机参数（focal_length、horizontal_aperture、width、height）计算内参矩阵。
        self._compute_intrinsic_matrices()
        # 由 pattern 生成相机系里的逐像素光线起点与方向。
        # ray_starts/ray_directions 形状均为 (num_envs, num_rays, 3)。
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
        )
        # num_rays = 分辨率 H×W（每像素一条光线）。
        self.num_rays = self.ray_directions.shape[1]
        # 存放每条光线命中点（世界系坐标）的缓冲。
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        # ---- 处理相机相对父 frame 的 offset（你配的 pos/rot）----
        # 用户给的 offset.rot 用的是某种约定（convention="world" 即 forward=+X, up=+Z），
        # 这里把它统一转成相机内部使用的约定，得到 offset 的四元数。
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        # offset 的旋转/平移广播到所有环境。
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

        # 传感器自身的世界位姿（每步更新）。
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self.device)

        # 世界系下的光线起点/方向（每步由相机位姿变换得到）。
        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        # 初始化分组碰撞所需的索引缓冲（每条光线的组 id、每个 mesh 属于哪个组等）。
        self._create_ray_collision_groups()

    def _update_ray_infos(self, env_ids: Sequence[int]):
        """更新光线信息缓冲（每步计算相机世界位姿，并把相机系光线变换到世界系）。"""

        # 1) 取父 frame（base_link）的世界位姿。
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        # 2) 父 frame 位姿 ⊕ offset = 传感器世界位姿。
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        # 3) 记录位姿（world 约定与 ros 约定各存一份，方便不同下游取用）。
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        self._data.quat_w_ros[env_ids] = quat_w

        # 4) 把相机系光线变换到世界系。
        # 光线起点：先旋转（相机系→世界系），再加平移（相机位置）。
        # 注意 ray_starts 是「相对相机原点」的偏移，所以要先转到世界方向再加 pos_w。
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        # 光线方向：只旋转、不平移（方向向量无位置）。
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """填充传感器数据缓冲（每步求交并写出各类深度/法向输出）。"""

        # 更新光线（相机位姿→世界系光线）
        self._update_ray_infos(env_ids)
        # 更新 mesh 变换（地面/机器人连杆等会动，需每步刷新其世界位姿）
        self._update_mesh_transforms(env_ids)

        # 取 mesh 的世界变换及其逆变换（逆变换用于把光线转到 mesh 局部系做求交）。
        mesh_transforms, mesh_inv_transforms = self._get_mesh_transforms_and_inv_transforms()

        # 取 warp mesh 的 device 引用（所有 mesh 共享同一块 warp 内存）。
        mesh_wp = [i for i in GroupedRayCaster.meshes.values()][0]
        # Warp kernel：并行对每条光线求交，返回命中点 / 欧氏距离 / 法向。
        self.ray_hits_w, ray_depth, ray_normal, _, _ = raycast_mesh_grouped(
            mesh_wp_device=mesh_wp.device,
            mesh_wp_ids=self._mesh_wp_ids,
            mesh_transforms=mesh_transforms,
            mesh_inv_transforms=mesh_inv_transforms,
            ray_group_ids=self._ray_collision_groups[env_ids],
            mesh_idxs_for_group=self._mesh_idxs_for_group,
            meah_idxs_slice_for_group=self._meah_idxs_slice_for_group,
            ray_starts=self._ray_starts_w[env_ids],
            ray_directions=self._ray_directions_w[env_ids],
            max_dist=self.cfg.max_distance * 2,  # 乘 2 是为了兼容「像平面距离」与「射线距离」两种语义的差异
            min_dist=self.cfg.min_distance,
            return_distance=True,
            return_normal=True,
        )
        assert ray_depth is not None
        assert ray_normal is not None

        # ---- 更新输出缓冲 ----
        if "distance_to_image_plane" in self.cfg.data_types:
            # 像平面深度：沿相机光轴（相机系 +X）的深度分量。
            # 先把「欧氏距离 × 单位方向」还原成世界系的命中向量，再逆旋回相机系，
            # 取第 0 分量（相机系 +X 光轴）即得到像平面深度。数据在相机系，故只取第一分量。
            distance_to_image_plane = (
                math_utils.quat_apply(
                    math_utils.quat_inv(self._data.quat_w_world[env_ids]).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * self._ray_directions_w[env_ids]),
                )
            )[:, :, 0]
            # 变换之后再做最大距离截断。
            if self.cfg.depth_clipping_behavior == "max":
                # "max"：超过最大距离的裁剪到 max_distance，NaN（未命中）也填成 max_distance。
                distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
            elif self.cfg.depth_clipping_behavior == "zero":
                # "zero"：超过最大距离或未命中的像素填 0。
                distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
            # 写回输出缓冲，reshape 成 (N, H, W, 1)。
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
                -1, *self.image_shape, 1
            )

        if "distance_to_camera" in self.cfg.data_types:
            # 到相机的欧氏距离（不是像平面深度），直接对 ray_depth 做裁剪。
            if self.cfg.depth_clipping_behavior == "max":
                ray_depth = torch.clip(ray_depth, max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                ray_depth[ray_depth > self.cfg.max_distance] = 0.0
            self._data.output["distance_to_camera"][env_ids] = ray_depth.view(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            # 命中点法向（世界系），reshape 成 (N, H, W, 3)。
            self._data.output["normals"][env_ids] = ray_normal.view(-1, *self.image_shape, 3)
