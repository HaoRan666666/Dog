# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 强化学习训练脚本。

用法:
    python scripts/rsl_rl/train.py --task BPX_Walk_Flat-v0 --num 1024 --max_iterations 10000
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# 本地工具导入
import cli_args  # isort: skip

# ── 命令行参数 ────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False,
                    help="训练时录制视频")
parser.add_argument("--video_length", type=int, default=200,
                    help="视频长度（步数）")
parser.add_argument("--video_interval", type=int, default=2000,
                    help="视频录制间隔（步数）")
parser.add_argument("--num_envs", type=int, default=None,
                    help="并行环境数量")
parser.add_argument("--task", type=str, default=None,
                    help="训练任务名称，如 BPX_Walk_Flat-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="RL 智能体配置入口点")
parser.add_argument("--seed", type=int, default=None,
                    help="随机种子")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="最大训练迭代次数")
parser.add_argument("--distributed", action="store_true", default=False,
                    help="多 GPU / 多节点分布式训练")
parser.add_argument("--export_io_descriptors", action="store_true", default=False,
                    help="导出 IO 描述符")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None,
                    help="Ray 集成自动配置，通常不需要手动设置")

# 附加 RSL-RL 和 AppLauncher 的命令行参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录制视频需要启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清除 sys.argv 中的 hydra 参数，避免影响后续解析
sys.argv = [sys.argv[0]] + hydra_args

# ── 启动 Isaac Sim ─────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 检查 RSL-RL 版本 ──────────────────────────────────────────────
import importlib.metadata as metadata
import platform
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

# ═══════════════════════════════════════════════════════════════════
# 以下代码在 Isaac Sim 进程内执行
# ═══════════════════════════════════════════════════════════════════

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401  注册官方任务
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

import Dog.tasks  # noqa: F401  注册自定义任务

# ── CUDA 性能优化 ─────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True   # 允许 TF32 加速矩阵乘法
torch.backends.cudnn.allow_tf32 = True         # 允许 cuDNN 使用 TF32
torch.backends.cudnn.deterministic = False     # 关闭确定性模式（更快）
torch.backends.cudnn.benchmark = False         # 不自动搜索最优算法（稳定）


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    """训练主函数：创建环境、加载策略、运行 PPO 训练循环。"""

    # ── 用命令行参数覆盖默认配置 ───────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # ── 设置随机种子 ───────────────────────────────────────��───────
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # CPU 不能做分布式训练
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # ── 多 GPU 分布式训练配置 ──────────────────────────────────────
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        # 不同 GPU 用不同种子保证探索多样性
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # ── 日志目录 ──────────────────────────────────────────────────
    # 根目录: logs/rsl_rl/{experiment_name}/
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 子目录: {时间戳}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # ── IO 描述符导出 ─────────────────────────────────────────────
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments."
        )

    env_cfg.log_dir = log_dir

    # ── 创建 Isaac Sim 环境 ────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    # 多智能体环境转单智能体
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 如果是恢复训练，找到之前的 checkpoint 路径
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # ── 视频录制包装 ──────────────────────────────────────────────
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ── RSL-RL 向量化环境包装 ─────────────────────────────────────
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 创建 PPO Runner ───────────────────────────────────────────
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # 记录 git 状态到日志（方便复现）
    runner.add_git_repo_to_log(__file__)

    # 恢复训练时加载之前的模型
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # ── 保存当前配置到日志 ─────────────────────────────────────────
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── 开始训练 ──────────────────────────────────────────────────
    # init_at_random_ep_len=True: 初始 episode 长度随机化，防止过拟合
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 训练结束，关闭仿真
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
