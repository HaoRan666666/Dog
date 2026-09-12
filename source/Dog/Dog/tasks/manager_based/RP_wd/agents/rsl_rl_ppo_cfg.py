# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass  # 导入 @configclass 装饰器，用于将类标记为可序列化的配置类

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
# 从 RSL-RL 桥接层导入三个基础配置类：
#   RslRlOnPolicyRunnerCfg — On-Policy 训练 Runner 的基类配置（控制训练循环、日志等）
#   RslRlPpoActorCriticCfg — Actor-Critic 网络结构配置（定义隐藏层、激活函数等）
#   RslRlPpoAlgorithmCfg   — PPO 算法超参数配置（学习率、折扣因子、裁剪参数等）


@configclass  # 将该类标记为 Isaac Lab 的配置类，支持 YAML 序列化/反序列化和 Hydra 命令行覆写
class RslRlPpoEncoderActorCriticCfg:
    """带深度图像编码器的 Actor-Critic 配置（消费 dict + (N,8,H,W) 深度图观测）。

    策略类指向 Dog 自带的 ``Dog.policies.encoder_actor_critic:EncoderActorCritic``，
    完全不依赖 robolab rsl_rl fork 的 vision 专有模块（方案 B）。
    """

    class_name: str = "Dog.policies.encoder_actor_critic:EncoderActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    activation: str = "elu"
    # 走 CNN 编码器的观测项（其余标量项直接 concat 进 MLP）
    actor_encoder_obs_groups: list[str] = ["depth_image"]
    critic_encoder_obs_groups: list[str] = ["depth_image"]
    # 深度编码器结构：与 parkour 一致（输入 (8, 48, 64) 多帧深度序列）
    encoder_cfg: dict = {
        "channels": [4],
        "kernel_sizes": [3],
        "strides": [1],
        "hidden_sizes": [256, 256],
        "output_size": 128,
        "paddings": [1],
        "nonlinearity": "ReLU",
        "use_maxpool": True,
        "last_activation": "ReLU",
    }


@configclass  # 将该类标记为 Isaac Lab 的配置类，支持 YAML 序列化/反序列化和 Hydra 命令行覆写
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):  # 继承 On-Policy Runner 基类，定义一个完整的 PPO 训练配置
    num_steps_per_env = 16  # 每个环境每轮采集的步数，总样本数 = num_envs × num_steps_per_env（如 4096×16=65536）
    max_iterations = 150  # 最大训练迭代次数（每轮用一批新样本做一次 PPO 更新），也是 PPO 总轮数
    save_interval = 500  # 模型保存间隔（每 50 轮保存一次 checkpoint），用于断点续训和后续评估
    experiment_name = "RP_wd_walk_flat"  # 实验名称，日志根目录为 logs/rsl_rl/rcdog_walk_flat/
    policy = RslRlPpoActorCriticCfg(  # Actor-Critic 策略网络配置
        init_noise_std=1.0,  # 初始动作噪声标准差（PPO 探索用），训练初期大噪声鼓励探索，后期逐渐衰减
        actor_obs_normalization=True,  # Actor 是否对观测做经验归一化（running mean/std），False 表示不使用
        critic_obs_normalization=True,  # Critic 是否对观测做经验归一化，False 表示不使用
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",  # 激活函数用 ELU（Exponential Linear Unit），比 ReLU 更平滑，训练更稳定
    )
    algorithm = RslRlPpoAlgorithmCfg(  # PPO 算法超参数配置
        value_loss_coef=1.0,  # Value 函数损失权重，1.0 表示 value 和 policy 损失同等重要
        use_clipped_value_loss=True,  # 是否对 value 损失也做裁剪（PPO clip），True 可防止 value 更新过大
        clip_param=0.2,  # PPO 裁剪参数 ε，新旧策略概率比被限制在 [1-ε, 1+ε] = [0.8, 1.2] 内
        entropy_coef=0.005,  # 熵奖励系数，鼓励策略保持一定随机性防止过早收敛到次优解
        num_learning_epochs=5,  # 每轮采集的数据重复训练 5 个 epoch（PPO 允许重复利用同一批数据）
        num_mini_batches=4,  # 将一批数据拆成 4 个 mini-batch 训练（减小显存、增加更新多样性）
        learning_rate=1.0e-3,  # Adam 优化器学习率 0.001
        schedule="adaptive",  # 学习率调度策略：adaptive 模式下当 KL 散度偏离 desired_kl 时自动调整学习率
        gamma=0.99,  # 折扣因子 γ，接近 1 表示重视远期回报，99 步后的回报仍有 ~37% 权重
        lam=0.95,  # GAE（Generalized Advantage Estimation）的 λ 参数，平衡偏差和方差，0.95 偏向低方差
        desired_kl=0.01,  # 目标 KL 散度，adaptive schedule 的参照值：KL 过大→降学习率，KL 过小→升学习率
        max_grad_norm=1.0,  # 梯度裁剪的最大范数，防止单步梯度爆炸，超过 1.0 则缩放回 1.0
    )


@configclass
class PPORunnerTerrainCfg(PPORunnerCfg):
    """台阶（地形）任务的 Runner 配置：与平地共享超参，仅用独立 experiment_name 区分日志目录。"""

    experiment_name = "RP_wd_walk_terrain"


@configclass
class PPORunnerPlatformCfg(PPORunnerCfg):
    """高台/平台任务的 Runner 配置：与平地共享超参，仅用独立 experiment_name 区分日志目录。"""

    experiment_name = "RP_wd_walk_platform"


@configclass
class PPORunnerBipedCfg(PPORunnerCfg):
    """双轮足（后腿站立）任务的 Runner 配置：与平地共享超参，仅用独立 experiment_name 区分日志目录。"""

    experiment_name = "RP_wd_walk_biped"


@configclass
class PPORunnerStairCfg(PPORunnerCfg):
    """深度相机上台阶任务的 Runner 配置：观测改为 dict + 深度图，改用编码器策略。"""

    experiment_name = "RP_wd_walk_stair"
    # 观测组映射：policy/critic 各自只用同名观测组（台阶观测已设为分项 dict 输出）
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    # 换成带深度图像编码器的 Actor-Critic（Dog 自带，不依赖 robolab rsl_rl fork）
    policy = RslRlPpoEncoderActorCriticCfg()


@configclass
class PPORunnerPlatformDepthCfg(PPORunnerCfg):
    """深度相机上高台任务的 Runner 配置：观测改为 dict + 深度图，改用编码器策略。"""

    experiment_name = "RP_wd_walk_platform_depth"
    # 观测组映射：policy/critic 各自只用同名观测组（高台深度观测已设为分项 dict 输出）
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    # 换成带深度图像编码器的 Actor-Critic（Dog 自带，不依赖 robolab rsl_rl fork）
    policy = RslRlPpoEncoderActorCriticCfg()


@configclass
class PPORunnerGapDepthCfg(PPORunnerPlatformDepthCfg):
    """跨沟独立实验；复用支持深度历史帧的 CNN Actor-Critic。"""

    experiment_name = "RP_wd_walk_gap_depth"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
