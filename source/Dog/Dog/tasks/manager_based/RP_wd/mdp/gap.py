"""跨沟任务的成功、失败与课程逻辑，坐标均相对当前地形出生原点。"""

import torch

from isaaclab.managers import SceneEntityCfg


def gap_goal_x(env):
    """各环境当前等级的远岸 + 落地余量，不能使用统一最大沟宽终点。"""
    terrain = env.scene.terrain
    cfg = terrain.cfg.terrain_generator.sub_terrains["gap"]
    level = terrain.terrain_levels.float().clamp(0, cfg.num_levels - 1)
    width = cfg.gap_width_range[0] + (
        cfg.gap_width_range[1] - cfg.gap_width_range[0]
    ) * level / (cfg.num_levels - 1)
    return cfg.approach_length + width + cfg.landing_clearance


def gap_success(env,
                sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*FOOT_LINK")):
    """到达远岸、姿态直立且至少两个轮足着地；腾空飞过终点不算成功。"""
    robot = env.scene["robot"]
    cfg = env.scene.terrain.cfg.terrain_generator.sub_terrains["gap"]
    position = robot.data.root_pos_w - env.scene.env_origins
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    grounded = (torch.linalg.vector_norm(forces, dim=-1) > 5.0).sum(dim=-1) >= 2
    return ((position[:, 0] >= gap_goal_x(env)) & (position[:, 1].abs() < cfg.lane_half_width)
            & (position[:, 2] > 0.25) & (robot.data.projected_gravity_b[:, 2] < -0.8)
            & grounded & ~gap_failure(env))


def gap_failure(env, minimum_height: float = 0.12):
    """落沟或离开通道；在绕到相邻 tile 之前终止。"""
    position = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    generator = env.scene.terrain.cfg.terrain_generator
    cfg = generator.sub_terrains["gap"]
    return ((position[:, 2] < minimum_height) | (position[:, 1].abs() >= cfg.lane_half_width)
            | (position[:, 0] < -cfg.spawn_x + cfg.end_margin)
            | (position[:, 0] > generator.size[0] - cfg.spawn_x - cfg.end_margin))


def gap_success_bonus(env):
    """RewardManager 会乘 dt，这里抵消以得到每次成功固定的终点奖励。"""
    return env.termination_manager.get_term("success").float() / env.step_dt


def gap_centerline_penalty(env):
    return (env.scene["robot"].data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]).square()


def gap_terrain_levels(env, env_ids):
    """按当前沟宽的成功判定升降一级；最高级保持，初始 reset 不变。"""
    active = env.episode_length_buf[env_ids] > 0
    success = env.termination_manager.get_term("success")[env_ids]
    terrain = env.scene.terrain
    # 阻止最高级升级，避免 TerrainImporter 默认随机回退到其他等级。
    can_promote = terrain.terrain_levels[env_ids] < terrain.max_terrain_level - 1
    terrain.update_env_origins(env_ids, success & active & can_promote, ~success & active)
    return env.scene.terrain.terrain_levels.float().mean()
