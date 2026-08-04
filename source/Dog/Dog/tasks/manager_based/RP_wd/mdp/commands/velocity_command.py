from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class TerrainSplitVelocityCommandCfg(UniformVelocityCommandCfg):
    """Two-range velocity command: pit terrains get forward-only, plane terrains get omnidirectional."""

    pit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Velocity ranges for pit terrain environments."""

    plane_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Velocity ranges for plane terrain environments."""

    pit_col_threshold: int = MISSING
    """Terrain columns 0..threshold-1 are pits, threshold.. are plane."""

    # 父类要求，实际不使用（_resample_command 已覆写，直接读 pit_ranges / plane_ranges）
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges()


class TerrainSplitVelocityCommand(UniformVelocityCommand):
    """Command generator that uses different velocity ranges based on terrain type.

    Reads ``env.scene.terrain.terrain_types`` to determine which envs are on pit vs plane terrain,
    then samples commands from the corresponding ranges. Heading commands only apply to plane envs.
    """

    cfg: TerrainSplitVelocityCommandCfg

    def __init__(self, cfg: TerrainSplitVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # split envs by terrain column: cols < threshold → pit, cols >= threshold → plane
        terrain_types = env.scene.terrain.terrain_types
        self.is_pit_env = terrain_types < cfg.pit_col_threshold

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        # split by terrain type
        pit_mask = self.is_pit_env[env_ids]
        plane_mask = ~pit_mask

        # pit terrain: forward-only, no heading
        pit_ids = env_ids[pit_mask]
        if len(pit_ids) > 0:
            n = len(pit_ids)
            self.vel_command_b[pit_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.pit_ranges.lin_vel_x)
            self.vel_command_b[pit_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.pit_ranges.lin_vel_y)
            self.vel_command_b[pit_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.pit_ranges.ang_vel_z)
            self.is_heading_env[pit_ids] = False

        # plane terrain: omnidirectional + heading
        plane_ids = env_ids[plane_mask]
        if len(plane_ids) > 0:
            n = len(plane_ids)
            self.vel_command_b[plane_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.plane_ranges.lin_vel_x)
            self.vel_command_b[plane_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.plane_ranges.lin_vel_y)
            self.vel_command_b[plane_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.plane_ranges.ang_vel_z)
            # heading only for plane envs
            if self.cfg.heading_command:
                self.heading_target[plane_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.plane_ranges.heading)
                self.is_heading_env[plane_ids] = torch.empty(n, device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # standing (shared across both terrain types)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
