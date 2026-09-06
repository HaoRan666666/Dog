# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Callable

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_split_reward(
    env: ManagerBasedRLEnv,
    base_func: Callable,
    pit_scale: float,
    plane_scale: float,
    base_params: dict | None = None,
    pit_col_threshold: int = 7,
    dynamic_gravity_threshold: float | None = None,
    plane_fallen_gate: bool = False,
    fallen_threshold: float = 0.5,
) -> torch.Tensor:
    """按地形类型分别缩放任意奖励函数。

    坑地形 (pit, 前 N 列) 缩放 ``pit_scale``，
    平地 (plane, 剩余列) 缩放 ``plane_scale``。

    用法示例::

        # 无额外参数的基础函数
        RewTerm(
            func=mdp.terrain_split_reward,
            weight=1.0,
            params={
                "base_func": mdp.flat_orientation_l2,
                "pit_scale": -0.1,
                "plane_scale": -2.5,
                "base_params": {},
            },
        )

        # 带额外参数的基础函数
        RewTerm(
            func=mdp.terrain_split_reward,
            weight=1.0,
            params={
                "base_func": mdp.wheel_vel_penalty,
                "pit_scale": 0.0,
                "plane_scale": -0.05,
                "base_params": {
                    "sensor_cfg": SceneEntityCfg(...),
                    "command_name": "base_velocity",
                    ...
                },
            },
        )

    Args:
        env: RL 环境实例。
        base_func: 基础奖励函数，签名为 ``(env, **base_params) -> (num_envs,)``。
        pit_scale: 坑地形 env 的缩放系数。
        plane_scale: 平地 env 的缩放系数。
        base_params: 透传给 ``base_func`` 的参数字典。
        pit_col_threshold: 地形列索引 < 该值的视为坑，>= 视为平地。
        dynamic_gravity_threshold: 若设置，则用 projected_gravity 的
            z 分量（机体直立度）替代地形列号来判断 pit/plane。
            机器人站直时 projected_gravity_z ≈ -1 → upright ≈ 1 → plane；
            爬行/扭曲时 projected_gravity_z 偏小 → upright < threshold → pit。
            推荐值 0.85~0.95。

    Returns:
        缩放后的奖励张量，形状 ``(num_envs,)``。
    """
    if base_params is None:
        base_params = {}
    # 手动解析嵌套的 SceneEntityCfg（manager 只会解析顶层的）。
    if not hasattr(terrain_split_reward, "_resolved_ids"):
        terrain_split_reward._resolved_ids = set()
    resolved = terrain_split_reward._resolved_ids
    for value in base_params.values():
        if isinstance(value, SceneEntityCfg) and id(value) not in resolved:
            value.resolve(env.scene)
            resolved.add(id(value))
    base_reward = base_func(env, **base_params)

    if dynamic_gravity_threshold is not None:
        # 用躯干直立度动态判断: 站直→plane(紧), 倾斜/爬行→pit(松)
        # projected_gravity_b[:, 2] ∈ [-1, 0], -1=完全直立, 0=水平
        upright = -env.scene["robot"].data.projected_gravity_b[:, 2]
        is_pit = upright < dynamic_gravity_threshold
    else:
        terrain_types = env.scene.terrain.terrain_types
        is_pit = terrain_types < pit_col_threshold

    scales = torch.where(is_pit, pit_scale, plane_scale)
    reward = scales * base_reward
    if plane_fallen_gate:
        reward = reward * (~plane_fallen_mask(env, pit_col_threshold, fallen_threshold)).float()
    return reward


def plane_fallen_mask(
    env: "ManagerBasedRLEnv",
    pit_col_threshold: int = 7,
    fallen_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回 (num_envs,) bool：是否「平地地形且翻倒」。

    ``upright = -g_z``（站立=1、水平=0、仰面=-1），``upright < fallen_threshold``（默认 60°）判翻倒。
    供 ``terrain_split_reward`` / ``flatness_split_reward`` 的 ``plane_fallen_gate`` 门控使用：
    翻倒起身需要「腿撑地」「关节大幅摆动」，这些动作会触发相应惩罚、与起身方向打架，
    故仅在 plane 且翻倒时清零这些干扰项；坑地形（terrain_types<7）is_plane 恒为 False，
    即便爬台大幅前倾也完全不受影响。
    """
    is_plane = env.scene.terrain.terrain_types >= pit_col_threshold
    upright = -env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    fallen = upright < fallen_threshold
    return is_plane & fallen


def is_flat_terrain(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    flatness_threshold: float = 0.05,
) -> torch.Tensor:
    """判断脚下地形是否平坦，返回 ``(num_envs,)`` 的 0/1 浮点开关量。

    用 height_scanner 命中点高度的**分位数范围（P95−P5）**衡量平坦度，
    而非 max-min 这种由单个极值点决定的判据，因此对「少数射线越出台沿」不敏感：
    - 平坦：中间 90% 命中点高度接近 → 1；
    - 坡面/90° 墙：命中点高度呈双峰（台面 vs 坑底）→ 0。

    该开关量与机身倾角本身无关（只看脚下地形几何），因此可复用到任意奖励，
    作为「站平才约束、爬行中放宽」的切换判据。
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]          # [num_envs, num_rays] 地形命中高度
    p5 = torch.quantile(hit_z, 0.05, dim=1)
    p95 = torch.quantile(hit_z, 0.95, dim=1)
    terrain_range = p95 - p5                       # 中间 90% 命中点的垂直跨度
    return (terrain_range < flatness_threshold).float()


def flatness_split_reward(
    env: "ManagerBasedRLEnv",
    base_func: Callable,
    flat_scale: float,
    nonflat_scale: float = 0.0,
    base_params: dict | None = None,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    flatness_threshold: float = 0.05,
    pit_col_threshold: int = 7,
    plane_fallen_gate: bool = False,
    fallen_threshold: float = 0.5,
) -> torch.Tensor:
    """按脚下地形平坦度切换任意奖励函数。

    平坦（``is_flat_terrain`` 为 1）时缩放 ``flat_scale``，
    不平坦（爬行/坡面/悬空）时缩放 ``nonflat_scale``。

    用法示例::

        # 站平才罚机身倾斜，爬行中不罚（专治爬上后翘头）
        RewTerm(
            func=mdp.flatness_split_reward,
            weight=1.0,
            params={
                "base_func": mdp.flat_orientation_l2,
                "flat_scale": -2.5,
                "nonflat_scale": 0.0,
                "base_params": {},
            },
        )

        # 带额外参数的基础函数
        RewTerm(
            func=mdp.flatness_split_reward,
            weight=1.0,
            params={
                "base_func": mdp.joint_position_penalty,
                "flat_scale": -0.03,
                "nonflat_scale": 0.0,
                "base_params": {
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_KENN_JOINT"]),
                    "stand_still_scale": 5.0,
                    "velocity_threshold": 0.3,
                },
            },
        )

    Args:
        env: RL 环境实例。
        base_func: 基础奖励函数，签名为 ``(env, **base_params) -> (num_envs,)``。
        flat_scale: 脚下地形平坦时的缩放系数。
        nonflat_scale: 脚下地形不平坦（爬行/坡面/悬空）时的缩放系数。
        base_params: 透传给 ``base_func`` 的参数字典。
        sensor_cfg: height_scanner 传感器配置。
        flatness_threshold: P95−P5 命中点高度范围 < 该值视为平坦。

    Returns:
        缩放后的奖励张量，形状 ``(num_envs,)``。
    """
    if base_params is None:
        base_params = {}
    # 手动解析嵌套的 SceneEntityCfg（manager 只会解析顶层的）。
    if not hasattr(flatness_split_reward, "_resolved_ids"):
        flatness_split_reward._resolved_ids = set()
    resolved = flatness_split_reward._resolved_ids
    for value in base_params.values():
        if isinstance(value, SceneEntityCfg) and id(value) not in resolved:
            value.resolve(env.scene)
            resolved.add(id(value))
    base_reward = base_func(env, **base_params)
    is_flat = is_flat_terrain(env, sensor_cfg, flatness_threshold)
    scales = torch.where(is_flat.bool(), flat_scale, nonflat_scale)
    reward = scales * base_reward
    if plane_fallen_gate:
        reward = reward * (~plane_fallen_mask(env, pit_col_threshold, fallen_threshold)).float()
    return reward


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def _step_command_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    vy_scale: float = 0.3,
    yaw_scale: float = 0.8,
) -> torch.Tensor:
    """横移/转向软门控，值 ∈ [0,1]；静止或纯前后运动时≈0。

    ``cmd[:, 1]`` 为横移线速度指令 vy，``cmd[:, 2]`` 为转向角速度指令 wz。
    只有横移/转向指令非零时才放开摆动腿相关约束，避免静止/纯前进时无意义踏步。
    """
    cmd = env.command_manager.get_command(command_name)
    vy_gate = torch.clamp(torch.abs(cmd[:, 1]) / vy_scale, 0.0, 1.0)
    yaw_gate = torch.clamp(torch.abs(cmd[:, 2]) / yaw_scale, 0.0, 1.0)
    return torch.maximum(vy_gate, yaw_gate)


def feet_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_clearance: float = 0.05,
    std: float = 0.02,
    wheel_radius: float = 0.1025,
    vy_scale: float = 0.3,
    yaw_scale: float = 0.8,
) -> torch.Tensor:
    """鼓励摆动腿达到合理抬腿高度（目标 0.05m），而非越高越好。

    离地高度以**轮子最下方**为基准：``轮轴(FOOT_LINK link 原点)高度 - wheel_radius``。
    FOOT_LINK 的 link 原点就是轮轴（MJCF 里 wheel 圆柱 size="0.1025 0.02"、中心即 body 原点），
    因此落地时 clearance≈0、抬起 5cm 时 clearance≈0.05。用 Gaussian 目标跟踪，
    支撑腿（clearance≈0）贡献≈0，只有抬起接近目标高度的摆动腿拿到奖励。
    仅在横移/转向时生效。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 轮轴（FOOT_LINK 的 link 原点）世界系高度 [N, n_feet]
    foot_axle_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    clearance = foot_axle_z - wheel_radius
    err = clearance - target_clearance
    reward = torch.sum(torch.exp(-0.5 * (err / std) ** 2), dim=1)
    reward = reward * _step_command_gate(env, command_name, vy_scale, yaw_scale)
    return reward


def leg_usage_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    ema_decay: float = 0.02,
    vy_scale: float = 0.3,
    yaw_scale: float = 0.8,
) -> torch.Tensor:
    """惩罚四条腿长期离地占比不均（防止只有固定两条腿参与踏步）。

    每条腿维护一个「当前是否离地」的 EMA（离地=1、着地=0），近似长期离地占比；
    再惩罚四条腿 EMA 的方差。不强制 trot / 对角 / gait phase，只要求四条腿参与度均衡。
    仅在横移/转向时生效。

    返回正数（方差），配合 config 里的负 ``weight`` 作为惩罚使用。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = (contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0).float()  # [N, n_feet]

    # 持久化 EMA 缓冲（挂在 env 上跨步保留，与 action_sync 的缓存做法一致）
    if not hasattr(env, "_leg_air_usage_ema"):
        env._leg_air_usage_ema = torch.zeros_like(in_air)
    ema = (1.0 - ema_decay) * env._leg_air_usage_ema + ema_decay * in_air
    # 新 episode 起步处重置 EMA
    new_ep = (env.episode_length_buf <= 1).unsqueeze(-1)
    ema = torch.where(new_ep, in_air, ema)
    env._leg_air_usage_ema = ema.detach()

    usage_var = ema.var(dim=1)  # [N]
    reward = usage_var * _step_command_gate(env, command_name, vy_scale, yaw_scale)
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    defined in :attr:`synced_feet_pair_names` to bias the policy towards a desired gait,
    i.e trotting, bounding, or pacing. Note that this reward is only for quadrupedal gaits
    with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Compute per-foot acceleration penalty for sound suppression.

    Penalize large vertical (z) accelerations when a foot is in contact with the ground.
    """

    asset = env.scene["robot"]

    # shape: (Nenv, Nbody, 6)
    body_acc = asset.data.body_acc_w

    # shape: (Nenv, Nfeet)
    foot_acc_z = body_acc[:, sensor_cfg.body_ids, 2]

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    #shape [num_envs, num_feet]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

    acc_penalty = (foot_acc_z ** 2) * in_contact.float()
    acc_penalty = torch.clamp(acc_penalty, max=50.0)

    penalty = acc_penalty.sum(dim=1)
    reward = penalty

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed < 1.5).float()

    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


# ── 侧倾奖励/终止 ────────────────────────────────────────────────

def side_tilt_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """仅惩罚侧倾（roll），不惩罚俯仰（pitch）。

    ``projected_gravity_b[:, 1]`` 是重力在 body Y 的分量，
    站直/爬台前倾时 ≈0，侧翻越大值越大。
    """
    return torch.square(env.scene["robot"].data.projected_gravity_b[:, 1])


def flat_on_flat_terrain(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    flatness_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """站在平坦地形上（平地/中间台阶/台面）时惩罚机身倾斜，坡面/爬行中返回 0。

    用 height_scanner 命中点高度的**分位数范围（P95−P5）**判断脚下地形是否平坦，
    而非 max-min 这种由单个极值点决定的判据，因此对「少数射线越出台沿」不敏感：
    - 平坦：中间 90% 命中点高度接近 → 罚 ``g_x²+g_y²``，把机身拉平；
    - 坡面/90° 墙：命中点高度呈双峰（台面 vs 坑底）→ 分位范围大 → 0。

    开关量是**地形几何**，与机身倾角本身无关，因此对 double_pit 的每一级台阶
    （以及最终台面）都生效，且不会像 ``dynamic_gravity_threshold`` 那样把
    「爬台倾斜」和「翘头」混为一谈。height_scanner 的 ``ray_alignment="yaw"``
    保证射线始终垂直向下，命中点 z 不受机身俯仰/侧倾影响。
    """
    is_flat = is_flat_terrain(env, sensor_cfg, flatness_threshold)
    asset = env.scene[asset_cfg.name]
    tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return -is_flat * tilt

# ── 终止条件 ──────────────────────────────────────────────────────

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


def stand_up_reward(
    env: "ManagerBasedRLEnv",
    pit_col_threshold: float = 7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """仅平地地形奖励站起；坑地形返回 0。

    ``upright = -g_z``（站立=1、水平=0、仰面=-1）直接奖励机身朝向，给翻转方向提供密集梯度。
    返回 ``upright - 1``，即仰面=-2 / 侧躺=-1 / 站立=0，**全程单调无断崖**：
    梯度始终指向「站直」，不会像旧版 ``fallen_threshold`` 硬阈值那样在 60° 处突然把奖励清零，
    从而避免策略停在「侧躺/斜躺」而不是继续站起。整体缩放交给 RewTerm 的 ``weight`` 调节。
    ``is_plane`` 门控保证 pit 爬台（爬台大幅前倾）完全不受影响；站立时奖励为 0，
    不给正常行走加常量分。
    """
    asset = env.scene[asset_cfg.name]
    is_plane = env.scene.terrain.terrain_types >= pit_col_threshold
    upright = -asset.data.projected_gravity_b[:, 2]
    return is_plane.float() * (upright - 1.0)


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
