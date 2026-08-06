"""Pit terrain configuration for RP_wd platform climbing training.

Uses MeshPitTerrainCfg: the robot starts on flat ground, walks into a pit (depression),
then must climb UP out of the pit. The upward climb trains the "上高台" skill without
the robot ever spawning on top of elevated terrain.

double_pit=True creates two concentric pits for progressive difficulty.
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=30.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # 双层坑：主要攀爬训练
        "platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.7,
            pit_depth_range=(0.1, 0.6),
            platform_width=1.5,
            double_pit=True,
        ),
        # 平地：全向移动
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
    },
)
