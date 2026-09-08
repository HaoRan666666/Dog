"""RP_wd 环境自定义终止条件。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def side_tilt(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    backward_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身侧翻或后倒超过阈值则终止，与爬台俯仰角无关。

    - 侧翻：|g_y| > sin(limit_angle)
    - 后倒：g_x > sin(backward_angle)，即机身向后倾超过 backward_angle
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    side = torch.abs(g[:, 1]) > torch.sin(torch.tensor(limit_angle))
    backward = g[:, 0] > torch.sin(torch.tensor(backward_angle))
    return torch.logical_or(side, backward)


def base_fallen(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身翻倒（相对竖直倾斜超过 90°）则终止，方向无关。

    ``g_z = projected_gravity_b[:, 2]``：站立时 ≈ -1，机身越过水平线后变正。
    ``g_z > threshold`` 统一覆盖侧翻、后倒、四脚朝天，且天然放行爬台时的大幅前倾。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    return g[:, 2] > threshold


def base_fallen_terrain_split(
    env: "ManagerBasedRLEnv",
    pit_threshold: float = 0.1,
    pit_col_threshold: float = 7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """仅坑地形翻倒即终止（保持原样）；平地地形翻倒不终止（留给起身）。"""
    terrain_types = env.scene.terrain.terrain_types
    g_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    fallen = g_z > pit_threshold
    is_pit = terrain_types < pit_col_threshold
    return fallen & is_pit
