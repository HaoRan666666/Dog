"""Step terrain configuration for L1 quadruped robot locomotion training.

This module defines a terrain generator config focused on step/stairs-type terrains.
It is designed for curriculum-based training where difficulty progressively increases
as the robot learns to traverse more challenging steps.
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

STEP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.02, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.02, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.02, 0.12),
            platform_width=2.0,
        ),
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.2,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=4,
                height=0.02,
                size=(0.3, 0.3),
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=20,
                height=0.12,
                size=(0.5, 0.5),
            ),
            platform_width=2.0,
        ),
    },
)
"""Step-focused terrain configuration for L1 quadruped locomotion training.

Terrain types:
  - **pyramid_stairs**: Ascending staircase pyramid centered on a flat platform.
  - **pyramid_stairs_inv**: Inverted (descending) staircase pyramid.
  - **boxes**: Random grid of raised boxes the robot must step over.
  - **repeated_boxes**: Scattered boxes that grow in number and height with difficulty.

Curriculum:
  Difficulty ranges from 0.0 (easiest, ~2 cm steps) to 1.0 (hardest, ~15 cm steps).
  The L1 robot stands at approximately 0.4 m, so the hardest steps are about 37.5%
  of standing height.
"""
