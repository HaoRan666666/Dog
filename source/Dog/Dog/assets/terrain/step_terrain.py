"""Step terrain configuration for L1 quadruped robot locomotion training.

This module defines a terrain generator config focused on step/stairs-type terrains.
It is designed for curriculum-based training where difficulty progressively increases
as the robot learns to traverse more challenging steps.
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
STEP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), #每个子地形区域大小
    border_width=20.0,
    num_rows=30,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.02, 0.30),
            step_width=0.4,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.65,
            step_height_range=(0.02, 0.30),
            step_width=0.4,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
    },
)
