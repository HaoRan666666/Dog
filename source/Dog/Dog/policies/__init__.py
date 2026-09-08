# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dog 自带的策略网络实现（不依赖 rsl_rl fork 专有的 vision 模块）。"""

from .encoder_actor_critic import EncoderActorCritic, Conv2dHeadModel

__all__ = ["EncoderActorCritic", "Conv2dHeadModel"]
