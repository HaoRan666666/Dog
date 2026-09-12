"""沿 +X 跨越的单沟地形；原点在沟前安全平台，地表高度为零。"""

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


def gap_terrain(difficulty: float, cfg: "GapTerrainCfg"):
    """离散等级单沟；两侧保留实体隔断，拼接后沟不会横向连通。"""
    if cfg.num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    # TerrainGenerator 在每行的 [row/N, (row+1)/N) 内随机采样。
    # 量化回行号，使每一级具有确定沟宽，且最后一级精确达到最大值。
    level = min(int(np.clip(difficulty, 0.0, 1.0) * cfg.num_levels), cfg.num_levels - 1)
    width = cfg.gap_width_range[0] + (
        cfg.gap_width_range[1] - cfg.gap_width_range[0]
    ) * level / (cfg.num_levels - 1)
    length, breadth = cfg.size
    near = cfg.spawn_x + cfg.approach_length
    far = near + width
    if not (0 < cfg.gap_width_range[0] <= cfg.gap_width_range[1]):
        raise ValueError("gap_width_range must be positive and ordered")
    if not (0 < cfg.spawn_x < near < far < length and cfg.gap_depth > 0):
        raise ValueError("Spawn, approach, gap and landing must fit inside the terrain")
    if not (0 < cfg.side_margin < breadth / 2):
        raise ValueError("side_margin must leave an open gap between the side banks")

    def box(x0, x1, top, thickness, y0=0.0, y1=breadth):
        mesh = trimesh.creation.box(extents=(x1 - x0, y1 - y0, thickness))
        mesh.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, top - thickness / 2))
        return mesh

    meshes = [
        box(0, near, 0.0, cfg.gap_depth + 0.2),
        box(far, length, 0.0, cfg.gap_depth + 0.2),
        box(near, far, -cfg.gap_depth, 0.2, cfg.side_margin, breadth - cfg.side_margin),
        box(near, far, 0.0, cfg.gap_depth + 0.2, 0.0, cfg.side_margin),
        box(near, far, 0.0, cfg.gap_depth + 0.2, breadth - cfg.side_margin, breadth),
    ]
    return meshes, np.array([cfg.spawn_x, breadth / 2, 0.0])


@configclass
class GapTerrainCfg(SubTerrainBaseCfg):
    function = gap_terrain
    gap_width_range: tuple[float, float] = (0.2, 1.0)
    gap_depth: float = 1.0
    spawn_x: float = 0.6
    approach_length: float = 0.9
    # 4m 地块保留 2.5m 横向沟口，为深度相机隔开相邻沟。
    side_margin: float = 0.75
    num_levels: int = 20
    landing_clearance: float = 0.65
    lane_half_width: float = 0.45
    end_margin: float = 0.35


GAP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=10.0,
    num_rows=20,
    num_cols=8,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={"gap": GapTerrainCfg(proportion=1.0)},
)
