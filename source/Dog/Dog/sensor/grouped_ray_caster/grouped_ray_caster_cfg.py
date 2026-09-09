"""分组光线投射器（GroupedRayCaster）的配置类与便捷工具。

- ``GroupedRayCasterCfg``：在 Isaac 官方 ``MultiMeshRayCasterCfg`` 之上扩展出的传感器配置，
  通过 ``class_type`` 绑定到实现类 :class:`GroupedRayCaster`。
- ``get_link_prim_targets``：一个便捷函数，把一串连杆名转成一批射线投射目标
  （``RaycastTargetCfg``），让相机能把机器人自身的连杆也当作「可被射线命中的 mesh」。
"""

from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from .grouped_ray_caster import GroupedRayCaster


@configclass
class GroupedRayCasterCfg(MultiMeshRayCasterCfg):
    """分组光线投射器的配置。"""

    class_type: type = GroupedRayCaster
    """配置对应的实现类。Isaac 的 configclass 机制在实例化传感器时用它来 new 出具体对象。"""

    min_distance: float = 0.0
    """射线投射的最小距离：小于该距离的命中会被忽略（用于剔除贴脸/自遮挡的假命中）。"""


def get_link_prim_targets(
    links: list[str],
    prefix: str = "/World/envs/env_.*/Robot/",
    suffix: str = "/visuals",
    is_shared=True,  # 是否假设所有环境共享同一份 mesh（几何相同，仅位姿不同）
    **kwargs: dict,
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """根据连杆名列表构建射线投射目标。

    每个连杆生成一条 ``RaycastTargetCfg``，其 ``prim_expr`` 由 ``prefix + link + suffix`` 拼成，
    例如 ``base_link`` → ``/World/envs/env_.*/Robot/base_link/visuals``。
    这样相机会把每个连杆的 visual mesh 当作独立的投射目标。

    Args:
        links: 连杆名列表（如 ``["base_link", "LF_ABAD_LINK", ...]``）。
        prefix: prim 路径统一前缀，默认带 ``env_.*`` 通配以匹配所有并行环境。
        suffix: prim 路径后缀，默认 ``/visuals``（可视 mesh）。
        is_shared: 是否所有环境共享同一份 mesh 几何。
        **kwargs: 透传给 ``RaycastTargetCfg`` 的其它参数。

    Returns:
        每个连杆一条 ``MultiMeshRayCasterCfg.RaycastTargetCfg`` 的列表。
    """
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{prefix}{link}{suffix}", is_shared=is_shared, **kwargs)
        for link in links
    ]
