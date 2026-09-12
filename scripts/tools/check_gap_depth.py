"""在 Isaac Lab 环境中验证跨沟任务 reset、深度张量和 CNN 动作推理。"""

import argparse
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import numpy as np

import Dog.tasks  # noqa: F401
from Dog.policies.encoder_actor_critic import EncoderActorCritic
from Dog.tasks.manager_based.RP_wd.agents.rsl_rl_ppo_cfg import PPORunnerGapDepthCfg
from Dog.assets.terrain.gap_terrain import gap_terrain
from Dog.tasks.manager_based.RP_wd.mdp.gap import gap_goal_x, gap_terrain_levels
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg("RP_wd_Walk_Gap_Depth", device=args.device, num_envs=args.num_envs)
    play_cfg = parse_env_cfg("RP_wd_Walk_Gap_Depth_Play", device=args.device)
    platform_cfg = parse_env_cfg("RP_wd_Walk_Platform_Depth", device=args.device)
    assert cfg.scene.terrain.terrain_type == play_cfg.scene.terrain.terrain_type == "generator"
    assert cfg.scene.num_envs == args.num_envs
    platform_rot = platform_cfg.scene.depth_camera.offset.rot
    train_rot = cfg.scene.depth_camera.offset.rot
    play_cfg.scene.depth_camera.offset.rot = (0.0, 1.0, 0.0, 0.0)
    assert cfg.scene.depth_camera.offset.rot == train_rot
    assert platform_cfg.scene.depth_camera.offset.rot == platform_rot == (1.0, 0.0, 0.0, 0.0)
    print("PASS: train/Play configuration loading and camera copy isolation", flush=True)
    generator = cfg.scene.terrain.terrain_generator
    gap_cfg = generator.sub_terrains["gap"].replace(size=generator.size)
    assert generator.size == (4.0, 4.0) and generator.num_rows == gap_cfg.num_levels == 20
    for level in range(20):
        expected = 0.2 + 0.8 * level / 19
        for offset in (0.01, 0.99):
            meshes, origin = gap_terrain((level + offset) / 20, gap_cfg)
            assert len(meshes) == 5 and all(mesh.is_watertight for mesh in meshes)
            np.testing.assert_allclose(meshes[1].bounds[0, 0] - meshes[0].bounds[1, 0], expected)
            np.testing.assert_allclose(origin, (0.6, 2.0, 0.0))
            # 沟口仍为 1.6m；两侧各 1.2m，为相机隔开相邻沟。
            np.testing.assert_allclose(meshes[2].bounds[:, 1], (1.2, 2.8))
            np.testing.assert_allclose([meshes[3].bounds[1, 2], meshes[4].bounds[1, 2]], 0.0)
    # 不依赖物理滚动的课程边界测试：初次 reset、最低级失败、普通升级/降级、最高级保持。
    terrain = SimpleNamespace(
        cfg=cfg.scene.terrain, terrain_levels=torch.tensor([0, 0, 5, 5, 19]), max_terrain_level=20,
    )
    updates = []
    terrain.update_env_origins = lambda ids, up, down: updates.append((up.clone(), down.clone()))
    test_env = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain), episode_length_buf=torch.tensor([0, 10, 10, 10, 10]),
        termination_manager=SimpleNamespace(get_term=lambda name: torch.tensor([False, False, True, False, True])),
    )
    gap_terrain_levels(test_env, torch.arange(5))
    assert updates[0][0].tolist() == [False, False, True, False, False]
    assert updates[0][1].tolist() == [False, True, False, True, False]
    terrain.terrain_levels = torch.arange(20)
    torch.testing.assert_close(gap_goal_x(test_env), 0.9 + torch.linspace(0.2, 1.0, 20) + 0.65)
    print("PASS: 20 exact gap widths, isolated side banks, per-level goals and curriculum bounds", flush=True)
    # 保留全部 20 行，确保几何等级与课程等级一致；仅缩减横向列数。
    cfg.scene.terrain.terrain_generator.num_cols = 2
    env = gym.make("RP_wd_Walk_Gap_Depth", cfg=cfg)
    try:
        obs, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        torch.testing.assert_close(
            robot.data.root_pos_w - env.unwrapped.scene.env_origins,
            robot.data.default_root_state[:, :3], atol=1e-5, rtol=0.0,
        )
        torch.testing.assert_close(robot.data.root_quat_w, robot.data.default_root_state[:, 3:7])
        torch.testing.assert_close(robot.data.root_vel_w, torch.zeros_like(robot.data.root_vel_w))
        runner_cfg = PPORunnerGapDepthCfg()
        policy_cfg = runner_cfg.policy.to_dict()
        policy_cfg.pop("class_name")
        policy = EncoderActorCritic(
            obs, runner_cfg.obs_groups, env.unwrapped.action_manager.total_action_dim, **policy_cfg,
        ).to(env.unwrapped.device).eval()
        with torch.inference_mode():
            for _ in range(args.steps):
                command = env.unwrapped.command_manager.get_command("base_velocity")
                assert ((command[:, 0] >= 0.6) & (command[:, 0] <= 1.0)).all()
                assert (command[:, 1:] == 0.0).all(), "Expected forward-only commands"
                depth = obs["policy"]["depth_image"]
                assert depth.shape == (args.num_envs, 8, 48, 64), depth.shape
                assert torch.isfinite(depth).all(), "Non-finite depth observation"
                actions = policy.act_inference(obs)
                assert torch.isfinite(actions).all(), "Non-finite actions"
                obs, reward, _, _, _ = env.step(actions)
                assert torch.isfinite(reward).all(), "Non-finite rewards"
        print(f"PASS: reset + {args.steps} steps, depth (N,8,48,64), CNN inference and rewards")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Kit 关闭可能直接退出进程，先输出异常，避免验证失败被隐藏。
        traceback.print_exc()
        raise
    finally:
        app.close()
