
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

import dataclasses
from enum import Enum

import attrs

from cosmos_predict2.conditioner import (
    BooleanFlag,
    ReMapkey,
    TextAttr,
    VideoConditioner,
)
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.auxiliary.text_encoder import (
    CosmosTextEncoderConfig,
)
from imaginaire.config import make_freezable
from imaginaire.constants import (
    CHECKPOINTS_DIR,
    CosmosPredict2Video2WorldFPS,
    CosmosPredict2Video2WorldModelSize,
    CosmosPredict2Video2WorldResolution,
    get_cosmos_predict2_video2world_tokenizer,
)
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class SolverTimestampConfig:
    nfe: int = 35
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False  # whether generate forward or backward timestamps


@make_freezable
@attrs.define(slots=False)
class CosmosGuardrailConfig:
    checkpoint_dir: str = CHECKPOINTS_DIR
    offload_model_to_cpu: bool = True
    enabled: bool = True


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames
    CHANNEL_CONCAT = "channel_concat"  # First few frames of the video are concatenated in the channel dimension

    def __str__(self) -> str:
        return self.value


@make_freezable
@attrs.define(slots=False)
class Video2WorldPipelineConfig:
    adjust_video_noise: bool
    conditioner: LazyDict[VideoConditioner]
    conditioning_strategy: str
    min_num_conditional_frames: int
    max_num_conditional_frames: int
    sigma_conditional: float
    net: LazyDict[MinimalV1LVGDiT]
    tokenizer: LazyDict[TokenizerInterface]
    guardrail_config: CosmosGuardrailConfig
    precision: str
    rectified_flow_t_scaling_factor: float
    rectified_flow_loss_weight_uniform: bool
    resize_online: bool
    resolution: str
    ema: EMAConfig
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder: CosmosTextEncoderConfig = attrs.field(factory=CosmosTextEncoderConfig)
    input_video_key: str = "video"
    input_image_key: str = "images"
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)


# Cosmos Predict2 Video2World 2B
_PREDICT2_VIDEO2WORLD_NET_2B = L(MinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
)

_PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_10FPS = Video2WorldPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(VideoConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.0,
            input_key=["obs/language_embedding"],
        ),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=1,
    max_num_conditional_frames=2,
    net=_PREDICT2_VIDEO2WORLD_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    rectified_flow_loss_weight_uniform=True,
    resize_online=False,
    resolution="480",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=6,
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_video2world_tokenizer(model_size="2B"),
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=True,
    ),
)


@dataclasses.dataclass(frozen=True)
class _Video2WorldPipelineConfig:
    model_size: CosmosPredict2Video2WorldModelSize
    resolution: CosmosPredict2Video2WorldResolution
    fps: CosmosPredict2Video2WorldFPS
    natten: bool = dataclasses.field(default=False, kw_only=True)


def get_cosmos_predict2_video2world_pipeline(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    resolution: CosmosPredict2Video2WorldResolution = "480",
    fps: CosmosPredict2Video2WorldFPS = 10,
) -> Video2WorldPipelineConfig:
    assert model_size == "2B"
    assert resolution == "480"
    assert fps == 10

    return _PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_10FPS
