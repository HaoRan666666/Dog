# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

# RP_wd 深度相机跨沟（训练与自动评估）
for suffix, cfg_name in (
    ("", "RP_wd_Walk_Gap_Depth_Env"),
    ("_Play", "RP_wd_Walk_Gap_Depth_Env_Play"),
):
    gym.register(
        id=f"RP_wd_Walk_Gap_Depth{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_env_gap_depth_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerGapDepthCfg",
        },
    )

##
# Register Gym environments.
##

# RP_wd 轮腿机器人平地行走（训练）
gym.register(
    id="RP_wd_Walk_Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:RP_wd_Walk_Flat_Env",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# RP_wd 轮腿机器人平地行走（Play）
gym.register(
    id="RP_wd_Walk_Flat_Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:RP_wd_Walk_Flat_Env_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# RP_wd 轮腿机器人地形行走（训练）
gym.register(
    id="RP_wd_Walk_Terrain",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_terrain_cfg:RP_wd_Walk_Terrain_Env",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerTerrainCfg",
    },
)

# RP_wd 轮腿机器人地形行走（Play）
gym.register(
    id="RP_wd_Walk_Terrain_Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_terrain_cfg:RP_wd_Walk_Terrain_Env_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# RP_wd 轮腿机器人高台/平台行走（训练）
gym.register(
    id="RP_wd_Walk_Platform",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_platform_cfg:RP_wd_Walk_Platform_Env",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerPlatformCfg",
    },
)

# RP_wd 轮腿机器人高台/平台行走（Play）
gym.register(
    id="RP_wd_Walk_Platform_Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_platform_cfg:RP_wd_Walk_Platform_Env_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# RP_wd 双轮足（只用两条后腿站立）平地行走（训练）
gym.register(
    id="RP_wd_Walk_Biped",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_biped_cfg:RP_wd_Walk_Biped_Env",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBipedCfg",
    },
)

# RP_wd 双轮足（只用两条后腿站立）平地行走（Play）
gym.register(
    id="RP_wd_Walk_Biped_Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_biped_cfg:RP_wd_Walk_Biped_Env_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBipedCfg",
    },
)

# RP_wd 深度相机上台阶（训练）
gym.register(
    id="RP_wd_Walk_Stair",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_stair_cfg:RP_wd_Walk_Stair_Env",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerStairCfg",
    },
)

# RP_wd 深度相机上台阶（Play）
gym.register(
    id="RP_wd_Walk_Stair_Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_stair_cfg:RP_wd_Walk_Stair_Env_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerStairCfg",
    },
)

# RP_wd 深度相机上高台/平台（训练）
gym.register(
    id="RP_wd_Walk_Platform_Depth",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_preceptive_platform_cfg:RP_wd_Walk_Platform_Depth_Env",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerPlatformDepthCfg",
    },
)

# RP_wd 深度相机上高台/平台（Play）
gym.register(
    id="RP_wd_Walk_Platform_Depth_Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_preceptive_platform_cfg:RP_wd_Walk_Platform_Depth_Env_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerPlatformDepthCfg",
    },
)
