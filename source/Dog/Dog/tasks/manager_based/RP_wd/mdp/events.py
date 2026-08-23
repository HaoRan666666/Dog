"""RP_wd 平台环境的复位事件。"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def _sample_plane_init_quat(
    n: int, device: str, default_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """按比例采样平地 env 初始姿态：40% 站立 / 40% 侧躺 / 20% 仰面。

    返回 ``(quats, fallen)``：站立类别用 ``default_quat``（面向默认朝向）；
    翻倒类别 yaw 在 [0, 2π) 随机。``fallen`` 为 bool，供调用方决定是否把 base 降到 fall_height。
    """
    u = torch.rand(n, device=device)
    cat = torch.zeros(n, dtype=torch.long, device=device)
    cat[u >= 0.4] = 1  # 0.4~0.8 → 侧躺 (40%)
    cat[u >= 0.8] = 2  # 0.8~1.0 → 仰面 (20%)

    sign = (torch.rand(n, device=device) < 0.5).float() * 2.0 - 1.0
    roll = torch.where(cat == 1, sign * (math.pi / 2), torch.zeros(n, device=device))
    pitch = torch.where(cat == 2, torch.full((n,), math.pi, device=device), torch.zeros(n, device=device))
    yaw = torch.rand(n, device=device) * 2.0 * math.pi
    fallen_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    standing = (cat == 0).unsqueeze(-1)
    quats = torch.where(standing, default_quat, fallen_quat)
    return quats, cat != 0


def reset_plane_to_fallen(
    env,
    env_ids: torch.Tensor,
    fall_height: float = 0.2,
    pit_col_threshold: float = 7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """坑地形 env 保持站立；平地地形 env 按比例初始化姿态。

    - pit env：保持默认站立姿态与高度（行为与原 reset_base 一致）。
    - plane env：40% 站立 / 40% 侧躺 / 20% 仰面；翻倒时 base 降到 fall_height。
    - 速度统一清零。
    """
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)
    is_plane = env.scene.terrain.terrain_types[env_ids] >= pit_col_threshold

    default_quat = root_states[:, 3:7]
    plane_quats, fallen = _sample_plane_init_quat(n, env.device, default_quat)
    quats = torch.where(is_plane.unsqueeze(-1), plane_quats, default_quat)

    # default_root_state 的 xyz 是相对 env_origin 的局部偏移（与 reset_root_state_uniform 一致），
    # 世界坐标 = 局部偏移 + env_origin。
    pos = root_states[:, 0:3].clone() + env.scene.env_origins[env_ids]
    # 只有「平地且翻倒」的 env 才把 base 降到 fall_height；平地站立与坑 env 保持默认高度
    fall_on_plane = is_plane & fallen
    pos[:, 2] = torch.where(
        fall_on_plane,
        env.scene.env_origins[env_ids, 2] + fall_height,
        pos[:, 2],
    )

    asset.write_root_pose_to_sim(torch.cat([pos, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)
