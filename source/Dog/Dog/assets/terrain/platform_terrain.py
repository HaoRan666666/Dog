"""High platform terrain configuration for RP_wd locomotion training.

This module defines a terrain generator config focused on elevated platform/box terrains.
The robot must learn to climb onto platforms of increasing height (0.3m ~ 0.8m).
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.6,
            box_height_range=(0.3, 0.8),
            platform_width=1.5,
            double_box=False,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),
    },
)
