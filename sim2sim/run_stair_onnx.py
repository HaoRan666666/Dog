"""跑一遍导出的台阶策略 ONNX（depth_encoder.onnx + actor.onnx），并和原始 torch checkpoint 对比数值。

用法：
    python run_stair_onnx.py --onnx_dir /home/rp/model_server/rpwd_0908_stair/exported \
        --ckpt /home/rp/model_server/rpwd_0908_stair/model_30000.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

from encoder_policy import _DEPTH_SHAPE, _SCALAR_OBS_DIM, load_stair_policy


def run_onnx(onnx_dir: str, scalar_obs: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
    """真实部署时的推理顺序：CNN 编码深度图 -> 和标量拼接 -> Actor MLP 出动作。"""
    encoder_sess = ort.InferenceSession(
        os.path.join(onnx_dir, "depth_encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    actor_sess = ort.InferenceSession(os.path.join(onnx_dir, "actor.onnx"), providers=["CPUExecutionProvider"])

    (depth_feature,) = encoder_sess.run(None, {"depth_image": depth_image})
    obs = np.concatenate([scalar_obs, depth_feature], axis=-1)
    (action,) = actor_sess.run(None, {"obs": obs})
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, default="/home/rp/model_server/rpwd_0908_stair/exported")
    parser.add_argument("--ckpt", type=str, default="/home/rp/model_server/rpwd_0908_stair/model_30000.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    scalar_obs = rng.normal(size=(1, _SCALAR_OBS_DIM)).astype(np.float32)
    depth_image = rng.uniform(0.1, 3.0, size=(1, *_DEPTH_SHAPE)).astype(np.float32)

    onnx_action = run_onnx(args.onnx_dir, scalar_obs, depth_image)

    device = torch.device("cpu")
    policy = load_stair_policy(args.ckpt, device)
    obs = {
        "policy": {
            "obs_flat": torch.from_numpy(scalar_obs),
            "depth_image": torch.from_numpy(depth_image),
        }
    }
    with torch.no_grad():
        torch_action = policy.act_inference(obs).numpy()

    max_diff = np.abs(onnx_action - torch_action).max()
    print("onnx action :", onnx_action)
    print("torch action:", torch_action)
    print(f"max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-4, "onnx 和 torch 推理结果对不上，检查导出流程"
    print("OK: onnx 导出的 CNN + Actor 和原始 checkpoint 数值一致")


if __name__ == "__main__":
    main()
