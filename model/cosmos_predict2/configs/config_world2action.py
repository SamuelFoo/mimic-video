# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import attrs
from torch import nn

from cosmos_predict2.configs.defaults.ema import EMAConfig
from imaginaire.config import make_freezable
from imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class SchedulerConfig:
    alpha: float
    beta: float
    num_denoising_steps: int


@make_freezable
@attrs.define(slots=False)
class World2ActionPipelineConfig:
    precision: str
    scheduler: SchedulerConfig
    net: LazyDict[nn.Module]
    ema: EMAConfig
    xattn_layer_idx: int
    # Optional: slice the video DiT's hidden_states to the first N latent
    # timesteps before feeding into the action decoder's cross-attention.
    # None = use full 16 timesteps (default, current behavior).
    xattn_video_prefix_length: int | None = None
