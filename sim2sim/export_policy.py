"""导出 RSL-RL checkpoint → TorchScript (含 normalizer)"""

import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='RSL-RL checkpoint path')
parser.add_argument('--output', type=str, default=None, help='Output .pt path')
args = parser.parse_args()

if args.output is None:
    args.output = args.checkpoint.replace('.pt', '_exported.pt')

device = 'cpu'
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
state_dict = ckpt['model_state_dict']

# ── 提取 normalizer ─────────────────────────────────────────────
obs_mean = state_dict["actor_obs_normalizer._mean"].squeeze(0)
obs_std = state_dict["actor_obs_normalizer._std"].squeeze(0)

# ── 提取 actor 权重 ─────────────────────────────────────────────
actor_state = {}
for k, v in state_dict.items():
    if k.startswith("actor."):
        actor_state[k[len("actor."):]] = v

# ── 构建含 normalizer 的完整推理模型 ────────────────────────────
class PolicyWithNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('obs_mean', obs_mean)
        self.register_buffer('obs_std', obs_std)
        self.actor = nn.Sequential(
            nn.Linear(159, 512), nn.ELU(alpha=1.0),
            nn.Linear(512, 256), nn.ELU(alpha=1.0),
            nn.Linear(256, 128), nn.ELU(alpha=1.0),
            nn.Linear(128, 16),
        )
        self.actor.load_state_dict(actor_state)

    def forward(self, obs):
        x = (obs - self.obs_mean) / self.obs_std.clamp(min=1e-6)
        return self.actor(x)

model = PolicyWithNormalizer()
model.eval()

# ── 导出 TorchScript ────────────────────────────────────────────
example = torch.randn(1, 159)
traced = torch.jit.trace(model, example)
traced.save(args.output)
print(f"Exported to {args.output}")

# 验证
loaded = torch.jit.load(args.output)
with torch.no_grad():
    out1 = model(example)
    out2 = loaded(example)
    diff = (out1 - out2).abs().max().item()
    print(f"Verification: max diff = {diff:.2e}")
