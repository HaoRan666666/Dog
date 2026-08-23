"""RP_wd 平台环境的复位事件。"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def _sample_fallen_quat(n: int, device: str) -> torch.Tensor:
    """等概率采样翻倒朝向：侧翻(roll≈±90°) / 前扑·后仰(pitch≈±90°) / 仰面(pitch≈180°)。"""
    mode = torch.randint(0, 3, (n,), device=device)  # 0 侧翻 / 1 前后倒 / 2 仰面
    sign = (torch.rand(n, device=device) < 0.5).float() * 2.0 - 1.0
    roll = torch.where(mode == 0, sign * (math.pi / 2), torch.zeros(n, device=device))
    pitch = torch.where(mode == 1, sign * (math.pi / 2), torch.zeros(n, device=device))
    pitch = torch.where(mode == 2, torch.full((n,), math.pi, device=device), pitch)
    yaw = torch.rand(n, device=device) * 2.0 * math.pi
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def reset_plane_to_fallen(
    env,
    env_ids: torch.Tensor,
    fall_height: float = 0.2,
    pit_col_threshold: float = 7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """坑地形 env 保持站立；平地地形 env 随机翻倒姿态。

    - pit env：保持默认站立姿态与高度（行为与原 reset_base 一致）。
    - plane env：翻倒朝向采样自 ``_sample_fallen_quat``，高度降到 ``fall_height``（贴地靠物理沉降）。
    - 速度统一清零。
    """
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)
    is_plane = env.scene.terrain.terrain_types[env_ids] >= pit_col_threshold

    fallen_quat = _sample_fallen_quat(n, env.device)
    default_quat = root_states[:, 3:7]
    quats = torch.where(is_plane.unsqueeze(-1), fallen_quat, default_quat)

    # default_root_state 的 xyz 是相对 env_origin 的局部偏移（与 reset_root_state_uniform 一致），
    # 世界坐标 = 局部偏移 + env_origin。
    pos = root_states[:, 0:3].clone() + env.scene.env_origins[env_ids]
    # plane env：把 base 降到 fall_height（相对地形表面），贴地靠物理沉降到翻倒姿态
    pos[:, 2] = torch.where(
        is_plane,
        env.scene.env_origins[env_ids, 2] + fall_height,
        pos[:, 2],
    )

    asset.write_root_pose_to_sim(torch.cat([pos, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)
