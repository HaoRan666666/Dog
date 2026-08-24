from __future__ import annotations

from dataclasses import MISSING, field
from typing import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


def _default_ranges():
    """Dummy ranges to satisfy parent validation; never used at runtime."""
    return UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
    )


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


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

    def _update_command(self):
        """Post-processes the velocity command with terrain-aware clipping.

        Overrides the parent to use terrain-specific angular velocity ranges for
        heading-based control, instead of the dummy ``ranges.ang_vel_z``.
        """
        # Compute angular velocity from heading direction (plane envs only)
        if self.cfg.heading_command:
            # heading envs are always on plane terrain; pit envs have is_heading_env=False
            heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if len(heading_ids) > 0:
                heading_error = math_utils.wrap_to_pi(
                    self.heading_target[heading_ids] - self.robot.data.heading_w[heading_ids]
                )
                self.vel_command_b[heading_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.plane_ranges.ang_vel_z[0],
                    max=self.cfg.plane_ranges.ang_vel_z[1],
                )
        # Enforce standing (zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        # 翻倒时保持正常速度指令（不清零）：让速度追踪奖励
        # （track_lin_vel/track_ang_vel）推着机器人翻身并继续运动，
        # 而不是在指令=0 时把「躺着静止」当满分来奖励。


@configclass
class TerrainSplitVelocityCommandCfg(UniformVelocityCommandCfg):
    """Two-range velocity command: pit terrains get forward-only, plane terrains get omnidirectional."""

    class_type: type = TerrainSplitVelocityCommand

    pit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Velocity ranges for pit terrain environments."""

    plane_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Velocity ranges for plane terrain environments."""

    pit_col_threshold: int = MISSING
    """Terrain columns 0..threshold-1 are pits, threshold.. are plane."""

    fall_gravity_threshold: float = -0.5
    """g_z 超过此阈值（机身倾斜 >60°）即视为翻倒，清零平地地形 env 的速度指令。"""

    # 父类要求，实际不使用（_resample_command 已覆写，直接读 pit_ranges / plane_ranges）
    ranges: UniformVelocityCommandCfg.Ranges = field(default_factory=_default_ranges)
