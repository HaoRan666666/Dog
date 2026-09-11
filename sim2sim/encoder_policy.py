"""台阶任务策略：EncoderActorCritic 精简移植 + 加载函数。

从 ``Dog/policies/encoder_actor_critic.py`` 原样搬过来（只把 TensorDict 类型
标注换成 dict，逻辑不动），这套架构和数值正确性已用官方导出的 actor 权重
逐位交叉验证过。这里只额外加 ``load_stair_policy()``，用真实 checkpoint
``/home/rp/model_server/rpwd_0908_stair/model_30000.pt`` 里的 state_dict 维度
反推出的超参数（见下方 encoder_cfg/hidden_dims 注释）构造网络后加载权重。
"""

from __future__ import annotations

from functools import reduce
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributions import Normal


def _resolve_nn_activation(act_name: str) -> nn.Module:
    act_dict = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "crelu": nn.CELU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
    }
    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()

        activation_mod = _resolve_nn_activation(activation)
        last_activation_mod = _resolve_nn_activation(last_activation) if last_activation is not None else None
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)
        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x)
        return x


class EmpiricalNormalization(nn.Module):
    def __init__(self, shape: int | tuple[int] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        return

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self._std + self.eps) + self._mean


class Conv2dHeadModel(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        hidden_sizes: list[int],
        output_size: int,
        paddings: list[int] | None = None,
        nonlinearity: str = "ReLU",
        use_maxpool: bool = False,
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in channels]
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(paddings)):
            raise ValueError("Conv2dHeadModel channel, kernel, stride, and padding lists must have the same length.")

        activation = getattr(nn, nonlinearity)
        in_channels = [image_shape[0], *channels[:-1]]
        conv_layers: list[nn.Module] = []
        for in_ch, out_ch, kernel_size, stride, padding in zip(in_channels, channels, kernel_sizes, strides, paddings):
            conv_stride = 1 if use_maxpool else stride
            conv_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=conv_stride, padding=padding),
                    activation(),
                ]
            )
            if use_maxpool and stride > 1:
                conv_layers.append(nn.MaxPool2d(stride))
        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            probe = torch.zeros(1, *image_shape)
            conv_out_size = int(self.conv(probe).numel())
        self.head = MLP(conv_out_size, output_size, hidden_sizes, nonlinearity.lower(), last_activation=last_activation)
        self.out_features = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(start_dim=1))


class EncoderActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: dict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        actor_encoder_obs_groups: tuple[str, ...] | list[str] = ("depth_image",),
        critic_encoder_obs_groups: tuple[str, ...] | list[str] | str | None = "shared",
        encoder_cfg: dict[str, Any] | None = None,
        critic_encoder_cfg: dict[str, Any] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        encoder_onnx_stems: dict[str, str] | None = None,
        encoder_onnx_sequential_idx: int = 0,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.encoder_onnx_stems = encoder_onnx_stems
        self.encoder_onnx_sequential_idx = encoder_onnx_sequential_idx
        self.obs_groups = obs_groups
        self.actor_encoder_obs_groups = list(actor_encoder_obs_groups)
        self.critic_encoder_obs_groups = critic_encoder_obs_groups
        self.actor_obs_group = obs_groups["policy"][0]
        self.critic_obs_group = obs_groups["critic"][0]

        encoder_cfg = encoder_cfg or {}
        critic_encoder_cfg = critic_encoder_cfg or encoder_cfg
        self.actor_encoders = self._build_encoders(
            obs, self.actor_obs_group, self.actor_encoder_obs_groups, encoder_cfg
        )

        if critic_encoder_obs_groups == "shared":
            self.critic_encoders = self.actor_encoders
            critic_encoder_groups = self.actor_encoder_obs_groups
        elif critic_encoder_obs_groups is None:
            self.critic_encoders = None
            critic_encoder_groups = []
        else:
            critic_encoder_groups = list(critic_encoder_obs_groups)
            self.critic_encoders = self._build_encoders(
                obs, self.critic_obs_group, critic_encoder_groups, critic_encoder_cfg
            )
        self._critic_encoder_obs_groups_resolved = critic_encoder_groups

        num_actor_obs = self._num_1d_obs(
            obs, self.actor_obs_group, self.actor_encoder_obs_groups
        ) + self._encoder_output_size(
            self.actor_encoders, self.actor_encoder_obs_groups
        )
        num_critic_obs = self._num_1d_obs(
            obs, self.critic_obs_group, critic_encoder_groups
        ) + self._encoder_output_size(
            self.critic_encoders, critic_encoder_groups
        )

        self.state_dependent_std = state_dependent_std
        self.actor = self._build_actor(num_actor_obs, num_actions, actor_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic = self._build_critic(num_critic_obs, critic_hidden_dims, activation)

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _build_actor(self, num_actor_obs, num_actions, actor_hidden_dims, activation) -> nn.Module:
        if self.state_dependent_std:
            return MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        return MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)

    def _build_critic(self, num_critic_obs, critic_hidden_dims, activation) -> nn.Module:
        return MLP(num_critic_obs, 1, critic_hidden_dims, activation)

    def _build_encoders(self, obs: dict, obs_group: str, component_names: list[str], cfg: dict[str, Any]) -> nn.ModuleDict:
        encoders = nn.ModuleDict()
        group_obs = obs[obs_group]
        for component_name in component_names:
            if len(group_obs[component_name].shape) != 4:
                raise ValueError(f"Encoder observation '{obs_group}.{component_name}' must have shape (N, C, H, W).")
            encoders[component_name] = Conv2dHeadModel(tuple(group_obs[component_name].shape[1:]), **cfg)
        return encoders

    def _num_1d_obs(self, obs: dict, obs_group: str, encoder_component_names: list[str]) -> int:
        num_obs = 0
        group_obs = obs[obs_group]
        for component_name, component_obs in group_obs.items():
            if component_name in encoder_component_names:
                continue
            if len(component_obs.shape) != 2:
                raise ValueError(f"Observation component '{obs_group}.{component_name}' must have shape (N, D).")
            num_obs += component_obs.shape[-1]
        return num_obs

    def _encoder_output_size(self, encoders: nn.ModuleDict | None, obs_groups: list[str]) -> int:
        if encoders is None:
            return 0
        return sum(encoders[obs_group].out_features for obs_group in obs_groups)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def act_inference(self, obs: dict) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        return self.actor(obs)

    def get_actor_obs(self, obs: dict) -> torch.Tensor:
        group_obs = obs[self.actor_obs_group]
        obs_list = [
            component_obs
            for component_name, component_obs in group_obs.items()
            if component_name not in self.actor_encoder_obs_groups
        ]
        obs_list.extend(
            self.actor_encoders[component_name](group_obs[component_name])
            for component_name in self.actor_encoder_obs_groups
        )
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: dict) -> torch.Tensor:
        group_obs = obs[self.critic_obs_group]
        obs_list = [
            component_obs
            for component_name, component_obs in group_obs.items()
            if component_name not in self._critic_encoder_obs_groups_resolved
        ]
        if self.critic_encoders is not None:
            obs_list.extend(
                self.critic_encoders[component_name](group_obs[component_name])
                for component_name in self._critic_encoder_obs_groups_resolved
            )
        return torch.cat(obs_list, dim=-1)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


# ── 加载 checkpoint ─────────────────────────────────────────────────────
# 超参数从 checkpoint state_dict 的实际张量维度反推确认（见开发过程中的核实）：
#   actor.0.weight (512,287) → num_actor_obs=287 = 159(标量,53×3history) + 128(深度编码)
#   actor_encoders.depth_image.conv.0.weight (4,8,3,3) → in_channels=8(帧数), out=4, k=3
#   actor_encoders.depth_image.head.0.weight (256,12288) → 12288=4×48×64 (无下采样,padding=1,stride=1)
#   actor_encoders.depth_image.head.{0,2,4} → hidden_sizes=[256,256], output_size=128
#   critic.0.weight (512,2771) → num_critic_obs=2771 = 159+128+2484(height_scan history,不参与部署推理)
_NUM_ACTIONS = 16
_SCALAR_OBS_DIM = 159  # 与 lab2mujoco2.py 的 build_history_obs 输出维度一致(53×3)
_DEPTH_SHAPE = (8, 48, 64)  # (num_output_frames, height, width)
_ENCODER_CFG = dict(
    channels=[4],
    kernel_sizes=[3],
    strides=[1],
    hidden_sizes=[256, 256],
    output_size=128,
    paddings=[1],
    nonlinearity="ReLU",
    use_maxpool=True,
    last_activation="ReLU",
)
_ACTOR_HIDDEN_DIMS = [512, 256, 128]
_CRITIC_HIDDEN_DIMS = [512, 256, 128]
# critic 除标量+深度外还有 height_scan，部署不用 critic，用占位维度凑够 2771 即可
# (只影响构造出的 critic MLP 形状是否和 checkpoint 一致，不影响 actor 推理结果)。
_CRITIC_DUMMY_DIM = 2771 - _SCALAR_OBS_DIM - _ENCODER_CFG["output_size"]


def load_stair_policy(ckpt_path: str, device: torch.device) -> EncoderActorCritic:
    """从原始训练 checkpoint 构造并加载台阶任务的 EncoderActorCritic，返回 eval 模式的策略。"""
    dummy_obs = {
        "policy": {
            "obs_flat": torch.zeros(1, _SCALAR_OBS_DIM),
            "depth_image": torch.zeros(1, *_DEPTH_SHAPE),
        },
        "critic": {
            "obs_flat": torch.zeros(1, _SCALAR_OBS_DIM + _CRITIC_DUMMY_DIM),
            "depth_image": torch.zeros(1, *_DEPTH_SHAPE),
        },
    }
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = EncoderActorCritic(
        obs=dummy_obs,
        obs_groups=obs_groups,
        num_actions=_NUM_ACTIONS,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=_ACTOR_HIDDEN_DIMS,
        critic_hidden_dims=_CRITIC_HIDDEN_DIMS,
        actor_encoder_obs_groups=["depth_image"],
        # 注意：不能用 "shared"——checkpoint 里 actor_encoders.* 和
        # critic_encoders.* 的权重实际不同（训练时是两个独立的 encoder），
        # 用 "shared" 会让 actor/critic 复用同一个 nn.ModuleDict 对象，
        # load_state_dict 按 state_dict 顺序 in-place 覆盖参数时，critic 的权重
        # 会把 actor 的权重覆盖掉，导致 actor 推理用的实际是 critic 的深度编码器
        # （已实测复现：覆盖后 conv.0.weight 和 checkpoint 里的
        # critic_encoders.depth_image.conv.0.weight 完全一致，和
        # actor_encoders.depth_image.conv.0.weight 对不上，直接导致 actor 输出
        # 数值爆炸到 1e12 量级）。传 list 让二者各自建独立模块，形状仍能对上
        # state_dict，但不会互相覆盖。
        critic_encoder_obs_groups=["depth_image"],
        encoder_cfg=_ENCODER_CFG,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    policy.to(device)
    return policy
