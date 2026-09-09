"""分组光线投射器（Grouped Ray Caster）。

这个类在 Isaac 官方 ``MultiMeshRayCaster`` 之上实现「**按环境分组的光线求交**」：

- 它能读取**多个** prim path（地面 + 机器人各连杆），并在每步投射线前刷新这些 mesh 的位姿。
- 最关键的是给每条光线和每个 mesh 都打一个「碰撞组 id」（= 环境编号），
  让 GPU 上 2000+ 并行环境里，每条光线**只打自己环境里的 mesh**，不会串到别的环境去。

工作原理（三层索引，全部展平成 1D 喂给 Warp kernel）：

1. ``_mesh_wp_ids``：每个 (env, mesh) 槽位对应 warp 内存里哪个真实 mesh 的 id。
2. ``_mesh_idxs_for_group``：每个 (env, mesh) 槽位对应 ``mesh_transforms`` 展平数组里的下标，
   满足 ``index = env_id * total_meshes + local_mesh_idx``。
3. ``_meah_idxs_slice_for_group``：每个组（env）在 ``_mesh_idxs_for_group`` 里的切片边界，
   ``group g 的 mesh 下标 = _mesh_idxs_for_group[slice(boundary[g], boundary[g+1])]``。

于是 kernel 里：某条光线的 group id → 用切片取出「该组允许命中的 mesh 下标」→ 只和这些 mesh 求交。
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import re

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster import MultiMeshRayCaster
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.sim.views import XformPrimView

from Dog.utils.warp.raycast import raycast_mesh_grouped

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg

# import logger
logger = logging.getLogger(__name__)


class GroupedRayCaster(MultiMeshRayCaster):
    """分组光线投射器：读取多个 prim path，并在投射线前持续刷新 mesh 位姿。"""

    cfg: GroupedRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)

    def _initialize_warp_meshes(self):
        # 先做父类的 mesh 初始化（建立 warp mesh 缓存、填充 _mesh_positions_w 等）。
        super()._initialize_warp_meshes()

        # 构建一个「展平 mesh id」张量 _mesh_wp_ids，它与展平后的 mesh 变换一一对应：
        #   第 (env, slot) 个元素 = 该槽位对应的 warp mesh 在缓存里的全局 id。
        # 后续 kernel 靠这个 id 去 warp 内存里取真实的三角形数据。
        total_meshes_per_env = self._mesh_positions_w.shape[1]
        mesh_wp_ids_tensor = torch.zeros(
            (self._num_envs, total_meshes_per_env),
            dtype=torch.int64,
            device=self._device,
        )

        mesh_idx = 0
        for target_cfg in self._raycast_targets_cfg:
            # 找出匹配该 prim_expr 的所有 prim（例如某连杆的 visuals）。
            prims = sim_utils.find_matching_prims(target_cfg.prim_expr)
            ids = []
            for prim in prims:
                prim_path = prim.GetPath().pathString
                # 把 "env_123" 归一化成 "env_0"，以查询共享的 mesh 缓存
                # （所有环境用的是同一份 mesh 几何，只是位姿不同）。
                prim_path_ = re.sub(r"env_\d+", "env_0", prim_path)
                assert prim_path_ in GroupedRayCaster.meshes, (
                    f"Mesh at prim path {prim_path} (casted to {prim_path_}) not found in the mesh cache"
                    f" {GroupedRayCaster.meshes.keys()}"
                )
                ids.append(GroupedRayCaster.meshes[prim_path_].id)

            ids_tensor = torch.tensor(ids, device=self._device, dtype=torch.int64)
            count = self._num_meshes_per_env[target_cfg.prim_expr]

            # 三种情况，把 mesh id 填进每个 env 的对应槽位：
            if len(ids) == 1:
                # 全局共享的单个 mesh（如地面）：所有环境复用同一个 id。
                mesh_wp_ids_tensor[:, mesh_idx] = ids_tensor[0]
            elif len(ids) == count:
                # 每个环境一个 mesh（共享同一份几何）：广播到所有环境。
                mesh_wp_ids_tensor[:, mesh_idx : mesh_idx + count] = ids_tensor.unsqueeze(0)
            elif len(ids) == self._num_envs * count:
                # 每个环境各自独立的 mesh：reshape 成 (num_envs, count) 逐个填。
                mesh_wp_ids_tensor[:, mesh_idx : mesh_idx + count] = ids_tensor.view(self._num_envs, count)
            else:
                logger.warning(f"Mismatch in mesh counts for {target_cfg.prim_expr}")

            mesh_idx += count

        # 展平成 1D，与展平后的 mesh 变换数组一一对应。
        self._mesh_wp_ids = mesh_wp_ids_tensor.flatten()

    def _initialize_rays_impl(self):
        # 先走父类的光线初始化（生成相机系光线等）。
        super()._initialize_rays_impl()
        # 额外创建分组碰撞所需的索引缓冲。
        self._create_ray_collision_groups()

    def _create_ray_collision_groups(self):
        """创建光线碰撞组与 mesh 下标索引。

        给定 ``s = slice(self._meah_idxs_slice_for_group[group_id], self._meah_idxs_slice_for_group[group_id+1])``，
        可得到该组允许命中的 mesh 下标列表 ``self._mesh_idxs_for_group[s]``。
        这些下标指向 ``mesh_transforms`` / ``mesh_inv_transforms`` / ``mesh_wp_ids``。

        注意：与父类不同，GroupedRayCaster 把所有 mesh 变换视为「展平」数组，
        用下标来标识某条光线应该命中哪些 mesh。
        """

        # 每条光线的组 id = 其环境编号。形状 (num_envs, num_rays)。
        self._ray_collision_groups = (
            torch.arange(self._num_envs, dtype=torch.int32, device=self._device).unsqueeze(1).repeat(1, self.num_rays)
        )

        # 每个 (env, mesh) 槽位在展平 mesh 数组里的下标，初始 -1（表示无效）。
        _mesh_idxs_for_group = torch.ones(
            (self._mesh_positions_w.shape[0], self._mesh_positions_w.shape[1]),
            dtype=torch.int32,
            device=self._device,
        ).fill_(-1)
        mesh_idx = 0
        total_meshes = self._mesh_positions_w.shape[1]
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            count = self._num_meshes_per_env[target_cfg.prim_expr]
            # 计算该组 mesh 的展平下标：
            #   index = env_id * total_meshes + (组内 mesh 局部编号) + mesh_idx
            # 形状 (num_envs, count)。
            indices = (
                torch.arange(self._num_envs, device=self._device).unsqueeze(1) * total_meshes
                + torch.arange(count, device=self._device).unsqueeze(0)
                + mesh_idx
            )
            _mesh_idxs_for_group[:, mesh_idx : mesh_idx + count] = indices.int()
            mesh_idx += count
        # 展平成 1D：(num_envs * (global_meshes + local_meshes_per_env))。
        self._mesh_idxs_for_group = _mesh_idxs_for_group.flatten(
            0, 1
        )  # (num_envs * (global_meshes + local_meshes_per_env))

        # 每个组的切片边界：(num_envs + 1)，组 g 的 mesh 下标是 [boundary[g], boundary[g+1])。
        _meah_idxs_slice_for_group = torch.arange(self._num_envs + 1, dtype=torch.int32, device=self._device)
        _meah_idxs_slice_for_group *= self._mesh_positions_w.shape[1]
        self._meah_idxs_slice_for_group = _meah_idxs_slice_for_group  # (num_envs + 1)

    def _update_mesh_transforms(self, env_ids: torch.Tensor | None = None):
        """更新指定环境的 mesh 位姿。

        Args:
            env_ids: 需要更新 mesh 位姿的环境 ID。
        """
        # 逐组更新 mesh 的位置与朝向。
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            # 若该目标配置不跟踪位姿（静态 mesh），跳过，只推进槽位下标。
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # 取目标 mesh 的世界位姿。
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            # 若该 prim 的「可视 prim」与「物理/碰撞 prim」之间存在偏移，则用偏移修正位姿
            # （mesh_offsets 里存的就是视觉 prim 相对物理 prim 的位姿偏移）。
            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:  # mesh 非全局共享，即每个环境各自有一份
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            # 写入展平的位置/朝向数组（朝向为 (w, x, y, z)）。
            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w  # (w, x, y, z)
            mesh_idx += count

    def _get_mesh_transforms_and_inv_transforms(self):
        """返回 mesh 的变换及其逆变换（用于把光线转到 mesh 局部系做求交）。"""
        # 把位置(3) + 四元数(4) 拼成 7 维，展平成 (num_envs * total_meshes, 7)。
        mesh_transforms = torch.concatenate(
            [self._mesh_positions_w, self._mesh_orientations_w],
            dim=-1,
        ).reshape(
            -1, 7
        )  # (num_envs * (global_meshes + local_meshes_per_env), 7) # (px, py, pz, qw, qx, qy, qz)
        # 计算逆变换：inv(T) = (inv(q) * (-p), inv(q))
        inv_q = math_utils.quat_inv(self._mesh_orientations_w)
        inv_p = math_utils.quat_apply(inv_q, -self._mesh_positions_w)
        mesh_inv_transforms = torch.concatenate(
            [inv_p, inv_q],
            dim=-1,
        ).reshape(
            -1, 7
        )  # (num_envs * (global_meshes + local_meshes_per_env), 7) # (px, py, pz, qw, qx, qy, qz)
        return mesh_transforms, mesh_inv_transforms

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """用当前 mesh 位姿更新光线投射缓冲，并更新指定环境（碰撞组）的命中点。

        Args:
            env_ids: 需要更新缓冲的环境 ID。
        """
        # 更新光线（相机位姿 → 世界系光线）
        self._update_ray_infos(env_ids)
        # 更新 mesh 位姿（地面/机器人连杆等会动）
        self._update_mesh_transforms(env_ids)

        # 取 mesh 变换与逆变换
        mesh_transforms, mesh_inv_transforms = self._get_mesh_transforms_and_inv_transforms()

        # 取 warp mesh 的 device 引用，调用分组求交 kernel
        mesh_wp = [i for i in GroupedRayCaster.meshes.values()][0]
        self._data.ray_hits_w[env_ids], _, _, _, _ = raycast_mesh_grouped(
            mesh_wp_device=mesh_wp.device,
            mesh_wp_ids=self._mesh_wp_ids,
            mesh_transforms=mesh_transforms,
            mesh_inv_transforms=mesh_inv_transforms,
            ray_group_ids=self._ray_collision_groups[env_ids],
            mesh_idxs_for_group=self._mesh_idxs_for_group,
            meah_idxs_slice_for_group=self._meah_idxs_slice_for_group,
            ray_starts=self._ray_starts_w[env_ids],
            ray_directions=self._ray_directions_w[env_ids],
            max_dist=self.cfg.max_distance,
            min_dist=self.cfg.min_distance,
        )
