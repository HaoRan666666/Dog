"""把台阶任务策略导出为 ONNX：CNN 深度编码器和 Actor MLP 各导出一个独立文件。

沿用 ``encoder_policy.load_stair_policy`` 从原始训练 checkpoint 构造并加载权重
（已与官方导出逐位交叉验证过），不需要启动 Isaac Sim。

导出后部署侧的推理顺序：
    1. depth_encoder.onnx：原始深度图 (1, 8, 48, 64) → 128 维特征（未归一化）
    2. 把该 128 维特征和标量观测（159 维）拼接成 287 维向量
    3. actor.onnx：287 维向量（内部先做经验归一化再过 MLP） → 16 维动作

用法：
    python export_stair_onnx.py --ckpt /home/rp/model_server/rpwd_0908_stair/model_30000.pt --out exported_onnx
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from encoder_policy import _DEPTH_SHAPE, _SCALAR_OBS_DIM, load_stair_policy


def export_stair_onnx(ckpt_path: str, out_dir: str, opset_version: int = 12) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")
    policy = load_stair_policy(ckpt_path, device)

    # 1) CNN 深度编码器
    depth_encoder = policy.actor_encoders["depth_image"]
    dummy_depth = torch.zeros(1, *_DEPTH_SHAPE, device=device)
    encoder_path = os.path.join(out_dir, "depth_encoder.onnx")
    torch.onnx.export(
        depth_encoder,
        dummy_depth,
        encoder_path,
        input_names=["depth_image"],
        output_names=["depth_feature"],
        opset_version=opset_version,
        dynamo=False,  # 走旧版 TorchScript 导出器，产出单文件 onnx（不生成外部 .onnx.data）
    )
    print(f"Exported CNN depth encoder to {encoder_path}")

    # 2) Actor：归一化 + MLP 打包成一个模块，输入是「标量拼接 + CNN 特征」的向量
    num_actor_obs = _SCALAR_OBS_DIM + depth_encoder.out_features
    actor_module = nn.Sequential(policy.actor_obs_normalizer, policy.actor)
    dummy_actor_in = torch.zeros(1, num_actor_obs, device=device)
    actor_path = os.path.join(out_dir, "actor.onnx")
    torch.onnx.export(
        actor_module,
        dummy_actor_in,
        actor_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"Exported actor MLP to {actor_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/rp/model_server/rpwd_0908_stair/model_30000.pt")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "exported_onnx"))
    args = parser.parse_args()
    export_stair_onnx(args.ckpt, args.out)
