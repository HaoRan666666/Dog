# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""兼容补丁：让官方 ``rsl_rl``（leggedrobotics/rsl_rl，不含 vision 专有模块）支持嵌套观测。

官方 rsl_rl 的 ``RolloutStorage`` 用 ``torch.zeros(T, *value.shape)`` 分配观测缓冲。
当观测组是「分项 dict」（``concatenate_terms=False``，如台阶任务的 ``depth_image``）时，
``value`` 是嵌套 ``TensorDict``，其 ``shape`` 只有 batch 维（``[num_envs]``），
于是分配出的缓冲丢掉特征维、与运行时嵌套观测结构不一致，最终在 ``add_transition``
的 ``copy_`` 处崩溃（``AttributeError: 'Tensor' object has no attribute '_get_str'``）。

本模块把 robolab fork 里专有的 ``_allocate_rollout_obs_buffer`` 移植进 Dog，
并给官方 ``RolloutStorage.__init__`` 打补丁，使训练/回放脚本无需依赖 fork 即可跑嵌套观测任务。
"""

from __future__ import annotations

import functools

import torch
from tensordict import TensorDict


def allocate_rollout_obs_buffer(
    obs_template: TensorDict, num_transitions_per_env: int, device: str
) -> TensorDict:
    """按 ``obs_template`` 的（含嵌套组）结构分配 rollout 观测缓冲。

    对叶子张量，在 batch 维前插入时间维 ``num_transitions_per_env``；对嵌套 ``TensorDict``
    递归处理并保留其 batch 维。这样 ``self.observations`` 与运行时 ``transition.observations``
    结构一致，``copy_`` 才能正确匹配。
    """

    def expand(node: TensorDict | torch.Tensor) -> TensorDict | torch.Tensor:
        if isinstance(node, torch.Tensor):
            return torch.zeros(num_transitions_per_env, *node.shape, device=device, dtype=node.dtype)
        children = {key: expand(value) for key, value in node.items()}
        return TensorDict(children, batch_size=torch.Size([num_transitions_per_env, *node.batch_size]))

    top = {key: expand(value) for key, value in obs_template.items()}
    return TensorDict(
        top,
        batch_size=torch.Size([num_transitions_per_env, *obs_template.batch_size]),
        device=device,
    )


def patch_rollout_storage() -> None:
    """给官方 ``RolloutStorage.__init__`` 打补丁，改用嵌套感知的观测分配。幂等。"""
    from rsl_rl.storage.rollout_storage import RolloutStorage

    if getattr(RolloutStorage, "_dog_nested_obs_patched", False):
        return

    orig_init = RolloutStorage.__init__

    @functools.wraps(orig_init)
    def patched_init(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape, device="cpu"):
        orig_init(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        # 原实现对嵌套 dict 观测会丢失特征维，这里按嵌套结构重新分配。
        self.observations = allocate_rollout_obs_buffer(obs, num_transitions_per_env, device)

    RolloutStorage.__init__ = patched_init
    RolloutStorage._dog_nested_obs_patched = True
